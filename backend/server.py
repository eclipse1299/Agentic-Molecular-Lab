"""HTTP API for the frontend (FastAPI). Run from the repo root:

    .venv/bin/python backend/server.py            # http://127.0.0.1:8000 (serves frontend/dist if built)

Everything shown in the UI is read from the files the lab already writes:
  data/trajectories/ours_<run_id>.csv   one row per oracle call
  data/chats/<run_id>.jsonl             agent calls, proposal outcomes, round / policy / adversary events
  data/results_<tag>.json               multi-seed evaluation results
UI-started runs additionally get data/runs/<run_id>.json (config + status). One run executes at a time, in a
worker thread, so the API stays responsive while the (CPU-bound) lab loop runs.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import chem_core
from domain import DATA
from eval.pmo_auc import top10_auc, top10_curve
from orchestrator import run_lab, summarize

CHATS, TRAJ, RUNS_DIR = DATA / "chats", DATA / "trajectories", DATA / "runs"
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
AGENT_ORDER = ["scout", "branch_a", "branch_b", "branch_c", "coordinator", "adversary", "evidence"]
NAME_RE = re.compile(r"^(?:(?P<arm>ours|ablated|adv|noadv)_)?(?P<br>[abc]+)(?:_(?P<sm>known|cold))?_seed(?P<seed>\d+)$")

app = FastAPI(title="Agentic Molecular Lab API")


# ------------------------------------------------------------------------------------------ run registry
@dataclass
class Handle:
    id: str
    cfg: dict
    status: str = "running"  # running | finished | stopped | error
    started: float = field(default_factory=time.time)
    error: str | None = None
    summary: dict | None = None
    stop: bool = False
    loop: asyncio.AbstractEventLoop | None = None
    approvals: dict = field(default_factory=dict)


HANDLES: dict[str, Handle] = {}
_lock = threading.Lock()


def _traj_path(run_id: str) -> Path:
    return TRAJ / f"ours_{run_id}.csv"


def _chat_path(run_id: str) -> Path:
    return CHATS / f"{run_id}.jsonl"


def _valid_id(run_id: str) -> str:
    known = run_id in HANDLES  # just started: the oracle hasn't created its CSV yet
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id) or not (known or _traj_path(run_id).exists()):
        raise HTTPException(404, f"unknown run {run_id!r}")
    return run_id


def _save_meta(h: Handle):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{h.id}.json").write_text(json.dumps(
        {"id": h.id, "cfg": h.cfg, "status": h.status, "started": h.started, "error": h.error,
         "summary": h.summary}, default=str))


def _meta(run_id: str) -> dict | None:
    h = HANDLES.get(run_id)
    if h:
        return {"id": h.id, "cfg": h.cfg, "status": h.status, "started": h.started, "error": h.error,
                "summary": h.summary}
    f = RUNS_DIR / f"{run_id}.json"
    if f.exists():
        m = json.loads(f.read_text())
        if m["status"] == "running":  # server restarted mid-run: the worker thread is gone
            m["status"] = "stopped"
        return m
    return None


def _config_from_name(run_id: str) -> dict:
    """Evaluation runs (eval/*.py) have no meta file; recover their setup from the run id."""
    m = NAME_RE.match(run_id)
    if not m:
        return {}
    return {"branches": m["br"], "adversary": m["arm"] in (None, "ours", "adv"), "seed_mode": m["sm"] or "known",
            "seed": int(m["seed"])}


def _read_traj(run_id: str) -> list[dict]:
    rows = []
    if not _traj_path(run_id).exists():
        return rows
    with _traj_path(run_id).open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"n": int(r["call_n"]), "smiles": r["smiles"], "score": float(r["score"]),
                             "branch": r["origin_branch"], "sa": float(r["sa"]) if r["sa"] else None,
                             "ad": float(r["ad_similarity"]) if r["ad_similarity"] else None,
                             "alerts": [a for a in r["alerts"].split(";") if a]})
            except (ValueError, KeyError):
                continue  # a row still being flushed
    return rows


_log_cache: OrderedDict[str, tuple[tuple, list]] = OrderedDict()


def _read_log(run_id: str) -> list[dict]:
    path = _chat_path(run_id)
    if not path.exists():
        return []
    st = path.stat()
    key = (st.st_size, st.st_mtime_ns)
    hit = _log_cache.get(run_id)
    if hit and hit[0] == key:
        _log_cache.move_to_end(run_id)
        return hit[1]
    recs = []
    with path.open() as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # half-written last line of a live run
    _log_cache[run_id] = (key, recs)
    while len(_log_cache) > 3:
        _log_cache.popitem(last=False)
    return recs


# ------------------------------------------------------------------------------------------ derived views
def _tokens(rec: dict) -> tuple[int, int]:
    u = rec.get("usage") or {}
    return int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))


def _outcomes_by_call(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in recs:
        if r["kind"] == "proposal_outcome":
            out.setdefault(r["call_id"], []).append(r)
    return out


def _call_summary(r: dict, outs: list[dict]) -> dict:
    tin, tout = _tokens(r)
    args = (r.get("output") or {}).get("args") or {}
    ok = (r.get("verdict") or {}).get("allowed", True)
    scored = [o for o in outs if o.get("oracle_score") is not None]
    reasons = Counter(o["gate_reason"] for o in outs if o.get("gate_reason"))
    agent = r["agent"]
    if not ok:
        title = f"Blocked: {(r['verdict'] or {}).get('reason')}"
    elif agent == "scout":
        title = (args.get("brief") or "(empty brief)")[:140]
    elif agent == "adversary":
        title = args.get("diagnosis") or "(no diagnosis)"
    elif agent == "evidence":
        title = f"ChEMBL check, {str(args.get('verdict', '?')).replace('_', ' ')}: {str(args.get('smiles') or '')[:70]}"
    elif outs:
        title = f"{len(outs)} proposals · {len(scored)} scored · {len(outs) - len(scored)} rejected"
    else:
        title = f"{len(args.get('proposals', []))} proposals"
    return {"call_id": r["call_id"], "round": r.get("round"), "agent": agent, "model": r.get("model"),
            "allowed": ok, "reason": (r.get("verdict") or {}).get("reason"), "title": title, "latency_ms": r.get("latency_ms"),
            "tokens_in": tin, "tokens_out": tout, "ts": r.get("ts"), "n_outcomes": len(outs), "n_scored": len(scored),
            "rejections": dict(reasons), "best": max((o["oracle_score"] for o in scored), default=None)}


def _feed_item(s: dict, r: dict, outs: list[dict]) -> dict:
    args = (r.get("output") or {}).get("args") or {}
    smiles, text = None, s["title"]
    if s["agent"] == "scout":
        text = args.get("brief") or text
    elif s["agent"] == "adversary" and s["allowed"]:
        text = f"{args.get('diagnosis', '')} Instruction: {args.get('instruction', '')}".strip()
    elif s["agent"].startswith("branch_") and s["allowed"]:
        why = ", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in s["rejections"].items())
        text = f"Proposed {s['n_outcomes']}: {s['n_scored']} reached the oracle" + (f", rejected: {why}." if why else ".")
        top = max((o for o in outs if o.get("oracle_score") is not None), key=lambda o: o["oracle_score"], default=None)
        if top:
            smiles, text = top["smiles"], text + f" Best scored {top['oracle_score']:.2f}."
    return {"agent": s["agent"], "round": s["round"], "ts": (s["ts"] or "")[11:19], "text": text, "smiles": smiles,
            "allowed": s["allowed"]}


def _beam(rows: list[dict], k: int = 10) -> list[dict]:
    best: dict[str, dict] = {}
    for r in rows:
        key = chem_core.canonical(r["smiles"]) or r["smiles"]
        if key not in best or r["score"] > best[key]["score"]:
            best[key] = r
    top = sorted(best.values(), key=lambda r: (-r["score"], r["n"]))[:k]
    return [{"rank": i + 1, "smiles": r["smiles"], "score": r["score"], "ad": r["ad"], "sa": r["sa"],
             "branch": r["branch"], "alerts": r["alerts"], "call_n": r["n"]} for i, r in enumerate(top)]


def _curve(rows: list[dict]) -> list[list]:
    """[calls, best-so-far, top-10 mean so far], thinned to ~250 points."""
    if not rows:
        return []
    n = len(rows)
    freq = max(1, n // 250)
    t10 = dict(top10_curve([r["smiles"] for r in rows], [r["score"] for r in rows], n, freq))
    out, best = [], 0.0
    for r in rows:
        best = max(best, r["score"])
        if r["n"] in t10:
            out.append([r["n"], round(best, 4), round(t10[r["n"]], 4)])
    return out


def _run_row(run_id: str) -> dict:
    meta = _meta(run_id)
    cfg = (meta or {}).get("cfg") or _config_from_name(run_id)
    tp = _traj_path(run_id)
    n = max(0, sum(1 for _ in tp.open()) - 1) if tp.exists() else 0
    status = meta["status"] if meta else "finished"
    return {"id": run_id, "source": "ui" if meta else "eval", "status": status, "calls": n,
            "budget": cfg.get("budget") or n, "branches": cfg.get("branches"), "adversary": cfg.get("adversary"),
            "seed_mode": cfg.get("seed_mode"), "llm": cfg.get("llm"), "seed": cfg.get("seed"),
            "updated": tp.stat().st_mtime if tp.exists() else (meta or {}).get("started", time.time())}


def _state(run_id: str) -> dict:
    meta = _meta(run_id)
    cfg = (meta or {}).get("cfg") or _config_from_name(run_id)
    rows, recs = _read_traj(run_id), _read_log(run_id)
    outs = _outcomes_by_call(recs)
    calls = [r for r in recs if r["kind"] == "agent_call"]
    events = [r for r in recs if r["kind"] == "event"]
    rounds = [e for e in events if e["event"] == "round"]
    triggers = [e for e in events if e["event"] == "adversary_trigger"]
    policy = [e for e in events if e["event"] == "policy"]
    plans = [e for e in events if e["event"] == "plan" and e.get("decided")]
    evidence = next((e.get("results", []) for e in reversed(events) if e["event"] == "evidence"), [])
    summaries = [_call_summary(r, outs.get(r["call_id"], [])) for r in calls]
    tin = sum(s["tokens_in"] for s in summaries)
    tout = sum(s["tokens_out"] for s in summaries)
    by_agent = {a: {"calls": 0, "denied": 0, "tokens": 0} for a in AGENT_ORDER}
    for s in summaries:
        d = by_agent.setdefault(s["agent"], {"calls": 0, "denied": 0, "tokens": 0})
        d["calls"] += 1
        d["denied"] += (not s["allowed"])
        d["tokens"] += s["tokens_in"] + s["tokens_out"]
    gate = Counter()  # Gatekeeper rejections by reason, plus oracle-step policy rejections
    for r in recs:
        if r["kind"] == "proposal_outcome" and r.get("gate_reason"):
            gate[r["gate_reason"]] += 1
    feed = [_feed_item(s, calls[i], outs.get(s["call_id"], [])) for i, s in enumerate(summaries[-9:], len(summaries) - len(summaries[-9:]))]
    for e in rounds[-3:]:  # the coordinator is code, not an LLM call: surface its quota decisions from the round events
        if e.get("quotas"):
            q = ", ".join(f"{k.replace('branch_', '').upper()} {v}" for k, v in e["quotas"].items())
            feed.append({"agent": "coordinator", "round": e["round"], "ts": (e.get("ts") or "")[11:19], "allowed": True,
                         "text": f"Quotas this round: {q}. Oracle calls used: {e['oracle_calls_used']}.", "smiles": None})
    feed.sort(key=lambda f: (f["ts"], f["round"] or 0))
    h = HANDLES.get(run_id)
    pending = [{k: v for k, v in a.items() if k not in ("fut",)} for a in (h.approvals.values() if h else [])]
    used = len(rows)
    status = meta["status"] if meta else "finished"
    return {
        "id": run_id, "status": status, "error": (meta or {}).get("error"), "summary": (meta or {}).get("summary"),
        "cfg": cfg, "used": used, "budget": cfg.get("budget") or used,
        "round": max((s["round"] or 0 for s in summaries), default=0),
        "tokens": {"input": tin, "output": tout, "total": tin + tout}, "llm": cfg.get("llm"),
        "beam": _beam(rows), "curve": _curve(rows),
        "auc": round(top10_auc([r["smiles"] for r in rows], [r["score"] for r in rows], max(used, 1)), 4) if rows else None,
        "rounds": [{k: e.get(k) for k in ("round", "oracle_calls_used", "best", "top10_mean", "sa_top10", "ad_top10",
                                          "scaffolds_in_beam", "quotas", "trigger")} for e in rounds],
        "triggers": [{k: e.get(k) for k in ("round", "fired", "flagged", "acted", "diagnosis", "instruction", "stats", "skipped")}
                     for e in triggers],
        "agents": by_agent, "gatekeeper": dict(gate), "feed": feed,
        "planner": dict(Counter(p["chosen"] for p in plans)),  # batches the planner decided between: exploit vs explore
        "evidence": evidence,  # ChEMBL checks of the final top hits (live runs)
        "policy": {
            "events": [{k: e.get(k) for k in ("round", "verdict", "policy", "agent", "tool", "reason", "approved", "smiles", "branch", "ts")}
                       for e in policy[-12:]],
            "ask": sum(1 for e in policy if e["verdict"] == "ASK"),
            "ask_approved": sum(1 for e in policy if e["verdict"] == "ASK" and e.get("approved")),
            "deny": sum(1 for e in policy if e["verdict"] == "DENY"),
            "agent_denied": sum(1 for s in summaries if not s["allowed"]),
            "oracle_rejected": gate.get("policy_rejected", 0), "recorded": bool(policy), "pending": pending},
    }


# ------------------------------------------------------------------------------------------ routes
class StartRun(BaseModel):
    budget: int = Field(500, ge=50, le=10000)
    branches: str = Field("abc", pattern="^[abc]{1,3}$")
    adversary: bool = True
    llm: str = Field("offline", pattern="^(offline|anthropic)$")
    seed_mode: str = Field("known", pattern="^(known|cold)$")
    seed: int = Field(0, ge=0, le=999)
    ask_human: bool = False  # False = AUTOPILOT semantics: log electrophile asks and auto-reject
    evidence: bool | None = None  # check the final top hits against ChEMBL; None = on for live runs, off for the mock


@app.get("/api/config")
def config():
    active = next((h.id for h in HANDLES.values() if h.status == "running"), None)
    return {"llm_available": bool(os.getenv("ANTHROPIC_API_KEY")), "active_run": active}


class KeyPayload(BaseModel):
    api_key: str


@app.post("/api/config/key")
def set_api_key(body: KeyPayload):
    key = body.api_key.strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        # Also persist to .env so future runs remember it
        env_file = Path(__file__).resolve().parents[1] / ".env"
        env_file.write_text(f"ANTHROPIC_API_KEY={key}\n")
    return {"ok": True, "llm_available": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/api/runs")
def list_runs():
    ids = {p.stem for p in CHATS.glob("*.jsonl")} | {p.stem for p in RUNS_DIR.glob("*.json")} | set(HANDLES)
    rows = [_run_row(i) for i in ids if _traj_path(i).exists() or i in HANDLES]
    return sorted(rows, key=lambda r: -r["updated"])


@app.post("/api/runs", status_code=201)
def start_run(body: StartRun):
    if body.llm == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "Live LLM mode needs ANTHROPIC_API_KEY in the server's environment.")
    with _lock:
        active = next((h for h in HANDLES.values() if h.status == "running"), None)
        if active:
            raise HTTPException(409, f"Run {active.id} is still running; stop it first.")
        run_id = "ui_" + time.strftime("%Y%m%d_%H%M%S")
        h = Handle(run_id, body.model_dump())
        HANDLES[run_id] = h
    _save_meta(h)
    threading.Thread(target=_worker, args=(h,), daemon=True).start()
    return {"id": run_id}


def _worker(h: Handle):
    cfg = h.cfg

    async def approver(reason: str, content: dict) -> bool:
        if h.stop or not cfg.get("ask_human"):
            return False
        loop = asyncio.get_running_loop()
        h.loop = loop
        aid = uuid.uuid4().hex[:8]
        fut = loop.create_future()
        h.approvals[aid] = {"id": aid, "status": "pending", "reason": reason, "branch": content.get("branch"),
                            "smiles": content.get("arguments", {}).get("smiles"), "fut": fut,
                            "alerts": content.get("arguments", {}).get("alerts", []), "created": time.time()}
        ok = await fut
        h.approvals[aid]["status"] = "approved" if ok else "rejected"
        return ok

    try:
        sess = asyncio.run(run_lab(
            budget=cfg["budget"], branches=cfg["branches"], llm=cfg["llm"], seed=cfg["seed"], verbose=False,
            adversary=cfg["adversary"], seed_mode=cfg["seed_mode"], run_id=h.id, approver=approver, evidence=cfg.get("evidence"),
            should_stop=lambda: h.stop))
        h.summary = summarize(sess)
        if sess.stalled and sess.stall_is_error:  # agent calls failed (e.g. a rejected API request): that is a failure, not "budget spent"
            h.status, h.error = "error", f"Stopped after {sess.oracle.calls} oracle calls: {sess.stall_reason}."
        else:
            h.status = "stopped" if h.stop else "finished"
    except Exception as e:  # surfaced in the UI rather than dying silently
        h.status, h.error = "error", f"{type(e).__name__}: {e}"
    _save_meta(h)


@app.get("/api/runs/{run_id}")
def run_state(run_id: str):
    return _state(_valid_id(run_id))


def _resolve(h: Handle, approval: dict, ok: bool):
    fut = approval["fut"]
    if h.loop and not fut.done():
        h.loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(ok))


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    h = HANDLES.get(run_id)
    if not h or h.status != "running":
        raise HTTPException(409, "run is not running")
    h.stop = True
    for a in list(h.approvals.values()):
        if a["status"] == "pending":
            _resolve(h, a, False)
    return {"ok": True}


class Decision(BaseModel):
    approve: bool


@app.post("/api/runs/{run_id}/approvals/{approval_id}")
def decide(run_id: str, approval_id: str, body: Decision):
    h = HANDLES.get(run_id)
    a = h.approvals.get(approval_id) if h else None
    if not a or a["status"] != "pending":
        raise HTTPException(404, "no such pending approval")
    _resolve(h, a, body.approve)
    return {"ok": True}


@app.get("/api/runs/{run_id}/calls")
def calls(run_id: str, round: int | None = None, agent: str | None = None, verdict: str | None = None,
          offset: int = 0, limit: int = Query(50, le=200)):
    _valid_id(run_id)
    recs = _read_log(run_id)
    outs = _outcomes_by_call(recs)
    all_ = [_call_summary(r, outs.get(r["call_id"], [])) for r in recs if r["kind"] == "agent_call"]
    items = [s for s in all_ if (round is None or s["round"] == round) and (agent is None or s["agent"] == agent)
             and (verdict is None or s["allowed"] == (verdict == "allowed"))]
    return {"total": len(items), "of": len(all_), "items": items[offset:offset + limit],
            "rounds": sorted({s["round"] for s in all_ if s["round"] is not None}),
            "agents": [a for a in AGENT_ORDER if any(s["agent"] == a for s in all_)]}


@app.get("/api/runs/{run_id}/calls/{call_id}")
def call_detail(run_id: str, call_id: str):
    _valid_id(run_id)
    recs = _read_log(run_id)
    rec = next((r for r in recs if r["kind"] == "agent_call" and r["call_id"] == call_id), None)
    if not rec:
        raise HTTPException(404, "no such call")
    outs = _outcomes_by_call(recs).get(call_id, [])
    return {**_call_summary(rec, outs), "system_prompt": rec.get("system_prompt"), "input": rec.get("input"),
            "output": rec.get("output"), "outcomes": [
                {k: o.get(k) for k in ("parent_id", "smiles", "gate_reason", "oracle_score", "ad_similarity", "sa_score", "alerts")}
                for o in outs]}


@app.get("/api/results")
def results():
    out = {}
    for tag, key in (("main", "known"), ("cold", "cold"), ("cold_live", "live")):
        f = DATA / f"results_{tag}.json"
        if f.exists():
            out[key] = json.loads(f.read_text())
    return out


@lru_cache(maxsize=512)
def _svg(smiles: str, w: int, h: int) -> str | None:
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = chem_core.mol_from_smiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.useBWAtomPalette()
    o.bondLineWidth = 1.6
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText().replace("#000000", "currentColor")


@app.get("/api/mol.svg")
def mol_svg(smiles: str, w: int = Query(240, ge=60, le=600), h: int = Query(160, ge=60, le=600)):
    svg = _svg(smiles, w, h)
    if svg is None:
        raise HTTPException(422, "invalid SMILES")
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# ------------------------------------------------------------------------------------------ static frontend
if DIST.exists():
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "no such endpoint")
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import argparse
    import os

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
