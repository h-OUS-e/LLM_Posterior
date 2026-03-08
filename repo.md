# LLM Posterior — Repo Structure

```
src/
  data_loader.py       IOPair, Task — load ARC JSON tasks
  grid_utils.py        grids_match, grid_diff, grid_to_str
  compress.py          compress_hypothesis — 4-line analyst summary
  hypothesis_agent.py  HypothesisOutput, generate_hypothesis (LOO/test calls)
  refine_agent.py      refine_hypothesis — spatial error feedback retry
  synthesize_agent.py  MasterOutput, synthesize_hypotheses — unify N analyst hypotheses
  generator_agent.py   GeneratorResult, generate_from_hypothesis — execute hypothesis on test input
  leave_one_out.py     HypothesisResult, TaskResult, run_task_loo — orchestrates full pipeline
  evaluate.py          evaluate_task, evaluate_batch — aggregate metrics

scripts/
  run_loo.py           CLI: --dry-run / --task-id / --num-tasks

tests/                 mirrors src/, one file per module

explore.ipynb          interactive: single-task visualization + batch evaluation (sections 1–9)
data/ARC-AGI/          ARC task JSON files (training + evaluation)
results/               JSON outputs from run_loo.py
```

## Pipeline

```
LOO splits (N concurrent)
  → refine wrong predictions (spatial feedback)
    → synthesize_agent (master hypothesis)
      → generator_agent (execute on test inputs)
```
