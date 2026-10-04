# Agentic Molecular Discovery Lab

A team of Claude agents that optimises molecules against the PMO **DRD2** oracle under a hard budget of oracle calls, then audits its own best results.

From five random ZINC molecules, the live lab reached a top-10 mean score of 0.9 after **45** and **133** oracle calls in two 300-call runs. Random sampling never got there within 300 calls. The lab then checks its winners three ways: a deterministic chemistry filter, an applicability-domain measure and a ChEMBL lookup. Those checks say the winners are probably classifier artifacts, and this README reports that finding next to the score.

## Results

Live runs: Haiku 4.5 for the Scout and three proposal branches, Sonnet 5.5 for the adversary, current agents, **cold start** (5 random ZINC molecules, no known ligands), budget **300 oracle calls**, adversary on.

| | Run 1 | Run 3 | Random ZINC (3 seeds, mean) | Single-call LLM (1 run) |
|---|---|---|---|---|
| Top-10 AUC at 50 calls | 0.285 | 0.043 | 0.026 | 0.097 |
| Top-10 AUC at 100 calls | 0.621 | 0.173 | 0.037 | n/a |
| Top-10 AUC at 250 calls | 0.832 | 0.617 | 0.064 | n/a |
| Top-10 AUC at 300 calls | **0.856** | **0.673** | 0.076 | n/a |
| Best DRD2 score | 0.989 | 0.966 | 0.36 to 0.74 | 0.570 |
| First score of 0.9 or more, at call | 29 | 110 | none in 300 | none in 51 |
| Top-10 mean of 0.9 or more, at call | **45** | **133** | none in 300 | none in 51 |
| Tokens used | 722k | 686k | 0 | not recorded |

Both runs finished with 0 oracle failures and 0 policy denials. Run IDs are `ui_20261004_132724` and `ui_20261004_152703`; their trajectories, chat logs and run summaries are in `backend/data/`.

**How to read this**

- **AUC** is the PMO top-10 AUC: the mean of the ten best unique scores, sampled every 10 calls, integrated and divided by the budget. Only checkpoints a run actually reached are shown. Nothing is padded.
- **Two completed runs, one starting beam.** Both runs start from the same five molecules (`--seed 0`), so they are repeats of one start, not two seeds. They differ a lot (AUC at 100 calls: 0.621 against 0.173), and two runs do not give a mean or a variance.
- **A third run** (`ui_20261004_151842`) was stopped by hand at 126 calls and is left out of the table. Over those 126 calls it reached a score of 0.9 at call 20 and a top-10 mean of 0.9 at call 37, best 0.996.
- **Random** is the mean of three ZINC sampling runs; the committed trajectories hold 500 calls each and the first 300 are used here. The single-call LLM is one prompt that yields about 50 molecules (51 were scored), so only its AUC at 50 calls is comparable. Neither is a like-for-like search baseline; see "What we do not claim".
- **Cost:** about 0.7M tokens per 300-call run. That is roughly $1 at Haiku 4.5 list prices, an estimate.

### Against the earlier agents

The first live runs used earlier agents: no recent-results memory and no surrogate-ranked selection, and the adversary call was rejected by the API, so it was effectively off. They ran 2 seeds for 500 calls; their run files are not in this repository, so these figures come from the earlier README. Their AUC at 250 calls was 0.130 and 0.671, and the first score above 0.5 came at call 305 and call 53. The current agents scored 0.832 and 0.617 at 250 calls, with the first score above 0.5 at calls 28 and 82. Two runs against two runs, with a spread bigger than the gap, does not separate the versions.

### Offline mock

The mock (`backend/agents/offline.py`) proposes RDKit edits with no LLM. It validates plumbing, policies, logging and evaluation, and says nothing about Claude's behaviour. Cold start, 3 seeds, budget 1,000: AUC 0.327 (0.290, 0.112, 0.581) against 0.182 for random (quoted from the earlier README; the run files are not committed). These mock numbers were generated before the plausibility filter was added, so they will not reproduce exactly. With the five known DRD2 ligands as seeds the oracle already scores about 1.0 on three of them, so warm-start numbers are not evidence of search quality.

## What the winners look like

The lab scores molecules with a classifier, so it also asks whether a high score means anything. The answer so far is that it does not yet.

- **Applicability domain.** The five known DRD2 ligands sit at AD similarity 0.51 to 0.70. The top-10 molecules of the two runs sit at **0.45** and **0.41**. Random starting molecules that score 0 sit at 0.39 to 0.47, so this measure has little range, but the winners are on the wrong side of it.
- **One series.** Run 1's top ten are a single fluorophenylbutanoyl diazepane and diazocane series that differs by halogen swaps on a heteroaryl ring. Run 3's top ten share a difluorophenylbutanoyl bicyclic-diamine core.
- **ChEMBL.** The final top five of each run were looked up in ChEMBL (target CHEMBL217, D2 receptor) at 70% similarity (run 1) and 50% (run 3). Every one came back `no_analogue`. The lookup works: haloperidol and aripiprazole through the same code return `analogue_active`, with measured pChEMBL up to 9.9 and 9.7. ChEMBL has nothing near these molecules.
- **Adversary.** Once a run succeeds, the trigger is active in most rounds, almost always on `high_score_low_domain`: 17 of 31 rounds in run 1 and 19 of 30 in run 3. Sonnet was called 5 and 4 times (the cooldown limits it). Eight of those nine calls diagnosed the leading series as likely classifier exploitation and told the flagged branch to move to known D2 pharmacophores; the first call in run 1 said it was not an exploit. There was no adversary-off arm, so we cannot say whether this changed the outcome.

No molecule here is claimed to bind DRD2.

## How it works

```
beam ──► Scout (Haiku): short SAR brief from the current beam
  ▲         │
  │         ▼
  │   Branch A  local substituent edits, no new rings      ┐
  │   Branch B  ring hopping, keep the pharmacophore       ├ 4 proposals each (Haiku)
  │   Branch C  bioisosteres from lower-ranked beam entries ┘
  │         │
  │         ▼
  │   Gatekeeper (code, no LLM): MW and logP caps, ring and scaffold rules,
  │   PAINS / BRENK alerts, free thiol and acyclic aminal filter, duplicates
  │         │ valid proposals
  │         ▼
  │   Surrogate (Tanimoto GP) ranks them; Planner picks exploit or explore
  │         │ best proposals up to the quota
  │         ▼
  └── DRD2 oracle: the only scorer, every call counted ──► scaffold-niched beam
                                   │
        telemetry digest (no raw SMILES) ──► deterministic trigger ──► Adversary (Sonnet)
                                                                          │
                          Coordinator: flagged branch quota cut to 1, instruction injected
```

- **Budget is the unit.** Rejected proposals cost no oracle call. The surrogate and planner decide which of the valid proposals spend the quota. The surrogate is a free Tanimoto-kernel GP and needs 30 scored molecules before it ranks anything; until then the agent's own order is kept.
- **Planner.** Per batch it compares *exploit* (rank by predicted mean) with *explore* (mean plus one standard deviation), by expected gain plus a learning term, per oracle call. Both options and the choice are logged.
- **Memory.** Branches and the Scout see the oracle's verdict on the last two rounds, their own fixable rejections and a short do-not-repeat list.
- **Adversary trigger.** It fires on low scaffold diversity, a rising SA trend, a falling AD-similarity trend, a high top-10 score at low AD (below the lowest known-ligand AD minus 0.05), or a composite gaming-divergence term. Thresholds were set by looking at the earlier live runs, so they are calibrated, not validated.
- **Evidence step.** At the end of a live run, the final top five are looked up in ChEMBL by an `evidence` specialist, through the same policy gate as everything else.

### Agents

| agent | model | job |
|---|---|---|
| scout | Haiku 4.5 | writes the SAR brief |
| branch_a, branch_b, branch_c | Haiku 4.5 | propose molecules |
| adversary | Sonnet 5.5 | only when triggered: a one-sentence exploit diagnosis and one corrective instruction, from a telemetry digest |
| coordinator | none (code) | quotas 4 / 4 / 4; a flagged branch drops to 1 (never 0), the rest is redistributed, two-round cooldown |
| evidence | none (tool only) | ChEMBL check of the final top five, with citations |

Agents are Omnigent `AgentDef`s run through Omnigent executors, with typed Omnigent `FunctionTool`s and `FunctionPolicy` objects. Omnigent's server and CLI runtime is not used: this repo's own loop (`backend/orchestrator.py`) drives the agents and holds the session state.

### Rules live in code, not prompts

Every tool call passes through an Omnigent `FunctionPolicy` before it runs.

| policy | verdict | rule |
|---|---|---|
| `write_permission` | DENY | an agent may call only its own tool; a proposal carrying a score field is denied, because only the oracle wrapper writes scores |
| `budget_cap` | DENY | `oracle_evaluate` is denied once the oracle-call ceiling is reached |
| `electrophile_approval` | ASK | a BRENK electrophile alert pauses for human approval in the UI; `AUTOPILOT=true` logs it and auto-rejects |
| `evidence_cap` | CAP | `lookup_chembl` takes one SMILES of at most 500 characters, runs at most 20 times per run, and only for the evidence agent |

In the two reported runs the gate allowed 429 and 424 calls and denied none; it asked 3 and 43 times about electrophiles, and all were auto-rejected.

## What we do not claim

- **No speed-up over Graph GA.** Graph GA's published DRD2 score (0.964 ± 0.012, [PMO, Gao et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/8644353f7d307baaf29bc1e56fe8e0ec-Paper-Datasets_and_Benchmarks.pdf)) is an AUC over a 10,000-call budget. Our AUC over 300 calls is a different quantity. PMO publishes no curve at 300 calls, and Graph GA has not been run on this oracle build here.
- **No claim that the LLM is needed.** There is no non-LLM search baseline at 300 calls, and the single-call control is one run.
- **No claim that the adversary helps.** It never ran in an on/off pair with a working Sonnet call.
- **The start is not prior-free.** The models know DRD2 pharmacology; the runs converge on butyrophenone-like series, and both runs descend from the same starting molecule.
- **One oracle.** DRD2 is a single SVM classifier. Nothing here shows the method carries over to other PMO tasks, regression oracles or real assays.
- **A score is not binding affinity.** Any hit would need experimental validation.
- **The gaming-divergence term is not independent evidence.** It is built from AD similarity and SA, not from a separate property predictor.
- **Token cost is measured; dollar cost is an estimate.**

## Run it

Python 3.13 (Omnigent needs 3.12 or newer). PyTDC 1.1.15 pins old scikit-learn and rdkit that do not build on 3.13, so it is installed without its pins, with a one-line shim for the removed `rdkit.six` (`backend/compat.py`). numpy must stay below 2.4: numpy 2.5 makes TDC's `float(array([x]))` raise, which TDC swallows into a silent 0.0 score. `oracle.py` runs an install self-test for exactly that.

```bash
git clone https://github.com/eclipse1299/Agentic-Molecular-Lab.git && cd Agentic-Molecular-Lab
python3.13 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt python-dotenv
.venv/bin/pip install --no-deps PyTDC==1.1.15
AUTOPILOT=true .venv/bin/python -m pytest -q          # 71 tests, about a minute
```

**Offline mock (no API key, free):**

```bash
AUTOPILOT=true .venv/bin/python backend/orchestrator.py --budget 100 --branches abc --seed-mode cold
AUTOPILOT=true .venv/bin/python backend/eval/run_seeds.py cold      # 3 seeds x {adversary on, off}
```

**Live Claude run** (the reported runs were started from the UI Setup page: budget 300, cold start, adversary on, seed 0):

```bash
export ANTHROPIC_API_KEY=...        # or paste it on the Setup page
.venv/bin/python backend/server.py  # http://127.0.0.1:8000
```

The Setup page can store the key in a local `.env` (gitignored). Keep the server on localhost. To aggregate finished runs into a results file: `backend/eval/live_results.py <tag> --ours <run ids> --budget 300`. `backend/eval/pmo_auc.py` pads short runs with their last value (the PMO convention), so for the numbers in this README use checkpoints a run actually reached.

**Frontend** (React 19, TypeScript, Tailwind, Vite). A built copy is in `frontend/dist`, so the server serves the UI as is. To rebuild: `cd frontend && npm install && npm run build`.

| page | shows |
|---|---|
| Home | headline numbers with their caveats, one rejected and one scored proposal |
| Setup | budget, branches, adversary, mock or live LLM, known or cold seeds, electrophile approval |
| Run | live oracle-call meter, beam depictions, score and AD/SA charts, agent feed, policy gate with human approve and deny, adversary banner, Stop |
| Inspector | every agent call: prompt, input, tool output, tokens, latency, verdict, and each proposal's outcome |
| Results | per-run AUC by budget checkpoint, with CSV export |

## Repository map

```
backend/
  orchestrator.py   session, main loop, adversary step, evidence step
  chem_core.py      Candidate, Gatekeeper, scaffold-niched beam, PAINS / BRENK, plausibility filter
  oracle.py         DRD2 wrapper: global call counter, failure log, trajectory CSV
  domain.py         applicability domain: the SVM's own 2,159 support vectors, ad_similarity
  surrogate.py      free Tanimoto-GP that ranks valid proposals
  planner.py        exploit versus explore, per batch
  telemetry.py      adversary digest (no raw SMILES) and the deterministic trigger
  chembl.py         read-only ChEMBL lookup of nearest neighbours' measured DRD2 activity
  policies.py       Omnigent FunctionPolicy objects and the gate
  llm.py            Anthropic executor (Haiku, Sonnet) and the offline mock executor
  agents/           scout, branch_a / b / c, adversary, coordinator, evidence, offline mock
  baselines/        random_stub.py, single_call_llm.py
  eval/             pmo_auc.py, ablation.py, run_seeds.py, live_results.py, render_results.py
  server.py         FastAPI: serves the UI, starts and stops runs, relays approvals
  tests/            71 tests
frontend/           React UI
```

**Provenance.** Per-call logs are in `backend/data/chats/<run_id>.jsonl` (system prompt, input, output, tokens, latency, policy verdict, and one record per proposal with its Gatekeeper reason or oracle score). Oracle trajectories are in `backend/data/trajectories/*.csv`, policy events in `backend/data/policy_events.jsonl`, run summaries in `backend/data/runs/`. Every number in the live-run results table can be recomputed from the committed trajectory CSVs. The offline-mock figures, the random baseline's AUC at 1,000 calls and the earlier-agents figures are quoted from earlier runs whose files are not committed.

## Next steps

1. Run seeds 1 and 2 live, so the starting beam differs.
2. Run an adversary-off arm on the same seeds.
3. Add a non-LLM baseline (Graph GA or a plain GA) on this oracle at 300 calls.
4. Replace the composite gaming-divergence term with an independent property predictor.
5. Test on a second oracle.

## Environment and licence

PyTDC 1.1.15, rdkit 2026.3.6, scikit-learn 1.9.1, numpy 2.3.5, Omnigent 0.16.0, anthropic 1.11 or newer. PMO and TDC scores have shifted across releases, so this oracle build may not score identically to the one Graph GA was measured on. MIT licence.
