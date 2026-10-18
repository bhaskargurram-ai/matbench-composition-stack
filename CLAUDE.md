# Matbench composition paper: project brief

## Owner and status
- **Owner:** Bhaskar Gurram (Independent Researcher). Paper email: gurrambhaskar.ai@gmail.com.
- **Repo:** private GitHub repo `bhaskargurram-ai/matbench-composition-stack`. It stays **private until the paper is submitted**. Do not make it public, and do not publish results anywhere, without the owner's explicit OK.
- **This is the owner's second paper.** The first was SIG-Net, submitted to JCIM on 2026-10-18 and public at github.com/bhaskargurram-ai/SIG-Net.

## Goal
Transfer the part of SIG-Net that was confirmed in pre-registered replication to the locked Matbench v0.1 composition tasks. That part is:
- cross-fitted stacking of diverse experts;
- followed by a novelty-damped local out-of-fold (OOF) residual correction (confirmed in SIG-Net with p = 0.016).

Run everything on CPU. Submit the results to the official Matbench leaderboard.

**Target journal:** Computational Materials Science (Elsevier), subscription track, $0 APC. Verify the journal's scope and current metrics before writing.

## Tasks (composition only; official 5-fold splits)

| Task | n | Metric | Current best on the official board (verify again before writing) |
|---|---|---|---|
| matbench_steels | 312 | MAE (MPa) | MODNet v0.1.12, 87.76 |
| matbench_expt_gap | 4,604 | MAE (eV) | Darwin 0.2865 (per-task board); MODNet 0.3327 |
| matbench_expt_is_metal | 4,921 | ROC-AUC | AMMExpress 0.9209 |
| matbench_glass | 5,680 | ROC-AUC | MODNet 0.9603 |

`matbench_jdft2d` is a **structure** task. It is out of scope.

## Honesty rules (non-negotiable; same standard as SIG-Net)
1. **Use the official folds only.** Use `matbench` `MatbenchBenchmark`, record via `task.record`, and never re-split.
2. **Never use test-fold data for fitting or selection.** All hyperparameter, expert and router choices are made with inner cross-validation inside each training fold. There is no separate validation set, so nested CV is the only legitimate selection route.
3. **Write the analysis plan before any test-fold result is viewed.** Write `ANALYSIS_PLAN.md`: primary endpoint, comparators, statistical tests, and reporting rules. Freeze the design and record SHA-256 hashes of the code before running the final evaluation.
   - Develop the design on inner folds of fold 0's training data only.
4. **Report everything,** including tasks where the method does not win. No cherry-picking of tasks, folds or seeds.
5. **Expect realistic margins:** a few percent versus the best leaderboard entry. Gains of 10–30% are plausible only against single baselines. Never claim a "guaranteed" margin.
6. **Verify every reference** (DOI, authors, year) before citing it. Do not fabricate retrieval records. ResearchStudio has the same rule.
7. **Never print or read secrets.** `.env` holds API credentials and is git-ignored.

## Method sketch (to be refined by the ResearchStudio idea stage, then frozen)
- **Featurisations:**
  - Magpie / matminer composition descriptors;
  - element-fraction vectors;
  - an optional learned representation (Roost/CrabNet-style, CPU-feasible).
- **Experts:**
  - gradient boosting (XGBoost/LightGBM) on Magpie features;
  - random forest;
  - kernel ridge / GP on element fractions;
  - MLP;
  - an optional CrabNet/Roost-type model;
  - MODNet, if it runs on CPU in reasonable time.
- **Cross-fitting:** 5-fold OOF inside each official training fold. Save OOF, inner and test predictions per expert.
- **Combiner:** simplex-constrained global stack (non-negative weights summing to 1).
- **Local correction:** novelty-damped local OOF residual correction. Neighbours come from a composition-space kernel, for example an element-fraction / Magpie-distance RBF. The correction strength and neighbourhood size are chosen by inner CV.
- **Classification tasks:** Brier-based stacking, plus a calibration check.
- **Ablations:**
  - single experts;
  - global stack without correction;
  - correction without stacking;
  - choice of kernel.
- **Statistics:** per-fold paired comparisons (5 folds per task, so report effect sizes and be clear about low power). Also report leaderboard rank.

## Workflow: ResearchStudio is the required research tool
The owner asked that the paper be developed **with microsoft/ResearchStudio**. Commit `063f100` is cloned in `_tools/ResearchStudio`.

ResearchStudio covers **ideation and prior-art checking** (Idea skills) and **post-paper assets** (Reel). It does **not** run experiments or write manuscripts. Those are done in this Claude Code session, which is the runtime ResearchStudio skills run on.

**Stage 1: idea and prior art (ResearchStudio-Idea; skills installed in `.claude/skills/`)**
1. `/paper-search`: Matbench composition-property ML, 2020–2026 (MODNet, CrabNet, Roost, Darwin, stacking/ensembles, residual kriging, local correction).
2. `/scoop-check`.
   - Problem: composition-based materials property prediction on small/medium data.
   - Claimed novelty: cross-fitted heterogeneous stacking with a novelty-damped local OOF residual correction in composition space.
3. `/idea-spark` with that direction and the constraint "CPU-only, official Matbench folds, honest nested CV". Record whether the idea card followed or departed from the stated direction, and why.
   - Keep run directories under `ideaspark_run/`.
4. Decide the final design. Write `ANALYSIS_PLAN.md`, then freeze it.

**Stage 2: experiments (CPU).** Use `.venv` (Python 3.12). Install experiment dependencies there: matbench, matminer, scikit-learn, xgboost, lightgbm, and torch-cpu if needed. Write one reproducible pipeline (a notebook plus modules, as in SIG-Net). Make it resumable, and save per-fold predictions.

**Stage 3: manuscript.** Elsevier `elsarticle` LaTeX. Generate all numbers and tables from result files, never by hand.
- Repo "Data and code availability" section: the repo URL becomes public only at submission.
- Include an Affiliation section ("Independent Researcher") and a declaration of competing interests.

**Stage 4: post-paper assets (ResearchStudio-Reel).** Poster, blog or video when needed. Reel needs poppler, LibreOffice and ffmpeg; its installer is bash-only, so install it in WSL (Ubuntu-24.04 is available) when this stage arrives.

## Environment notes (Windows)
- The ResearchStudio skills call `python3`. On this machine use `.venv\Scripts\python` (Python 3.12) instead.
- IdeaSpark **requires OpenReview credentials** (`OPENREVIEW_USER` and `OPENREVIEW_PASS` in `.env`, copied from `.env.template`). The owner must fill these in.
  - OpenAlex, Semantic Scholar and Exa keys are optional.
  - Never echo `.env`.
- ResearchStudio's `install.sh` refuses to run in Git Bash, so the Idea skills were installed manually: skill folders copied into `.claude/skills/`, and Python dependencies (feedparser, openreview-py, beautifulsoup4, pymupdf, requests) installed into `.venv`.
- Git commits end with the attribution lines given by the session's system reminder.
