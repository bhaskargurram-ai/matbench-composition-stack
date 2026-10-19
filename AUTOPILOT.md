# Autopilot instructions (headless session)

You are running **unattended**. Nobody can answer questions. Read `CLAUDE.md` first and follow it exactly, especially the honesty rules.

## What to do
Work through the stages in `CLAUDE.md` in order, as far as you can get:

### Stage 1 (ResearchStudio-Idea)
- Run `/paper-search`, then `/scoop-check`, then `/idea-spark`, using the skills in `.claude/skills/`. Use `.venv\Scripts\python` wherever a skill says `python3`.
- There is no `.env` with OpenReview credentials yet:
  - run the connectors that work without keys (arXiv, OpenAlex, Semantic Scholar, DBLP, Crossref);
  - record the reduced coverage in `STATUS.md`;
  - do not fabricate any retrieval records.
- Save the outputs under `ideaspark_run/` and `research/`. Write a one-page `research/STAGE1_SUMMARY.md` covering:
  - the closest prior work;
  - the scoop-check verdict;
  - the idea card;
  - the final design decision, and whether it followed or departed from the stated direction, and why.

### Stage 2 (experiments, CPU only)
1. Install the experiment dependencies into `.venv`.
2. Load the 4 Matbench composition tasks with the official folds.
3. Build the pipeline. Develop the design using **inner CV on fold 0's training data only**.
4. Write `ANALYSIS_PLAN.md`, then freeze the design: record SHA-256 hashes of the code in `results/freeze_hashes.txt` and commit them.
5. Only after the freeze, run the full evaluation on all official folds.
6. Save per-fold predictions and metrics.
7. Prepare the Matbench leaderboard submission files, but **do not submit anything**.

### Stage 3 (manuscript)
Write a complete draft in Elsevier `elsarticle` LaTeX under `manuscript/`:
- every number generated from the result files;
- references verified;
- figures included.

Compile it to PDF.

## Rules for running unattended
- **Stay inside this folder** (the repository root, `Matbench_Paper/`).
  - Do not touch other projects.
  - Do not delete anything outside this folder.
- **Git:**
  - Commit and push to the **private** `origin` after each milestone.
  - **Never change the repo's visibility,** never create public repos or gists, never publish anything, and never submit to the leaderboard.
- **Secrets:** never read, print or commit secrets.
- **Keep `STATUS.md` up to date** after every meaningful step. It should show:
  - the current stage;
  - what was done;
  - key numbers (marked as inner-CV vs. final);
  - problems and blockers;
  - the next step.
- **Don't spin on blockers.** If something blocks progress (missing credentials, a package that won't install, a model too slow on CPU):
  - write it in `STATUS.md`;
  - choose the most honest fallback;
  - continue.
  
  Never fake a result.
- **Keep runs bounded.** Long runs must be resumable and save to disk per fold. Prefer CPU-feasible models. Skip a model if one fold takes more than about 30 minutes, and record that you skipped it.
- **When you stop** (finished, or out of steps), leave `STATUS.md` with a precise "Next step" so that `claude -p --continue` can resume.
