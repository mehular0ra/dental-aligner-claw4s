# Dental Aligner RL Environment — Agent Guide

## Quick Start
```bash
uv sync && uv run python -m server.app &
sleep 3 && curl -s http://localhost:7860/health
```

## Two Modes

### Original (1-shot): `/reset` + `/step`
Agent submits full 24-stage trajectory at once.

### Stepwise (24-step): `/reset_stepwise` + `/step_stepwise` + `/tool`
Agent plans one stage at a time with tool-use actions.

## Stepwise API

**Reset:** `POST /reset_stepwise {"task_id":"task_easy","seed":42,"source":"synthetic"}`

**Step:** `POST /step_stepwise {"episode_id":"...","poses":[[qw,qx,qy,qz,tx,ty,tz]x28]}`

**Tools:** `POST /tool {"episode_id":"...","tool":"inspect_tooth","args":{"tooth_id":31}}`

Available tools: `inspect_tooth`, `simulate_step`, `check_collisions`, `commit_stage`, `rollback_stage`

## Key Constraints
- 28 teeth, 24 stages, poses as `[qw,qx,qy,qz,tx,ty,tz]`
- Max 0.25mm translation / 2.0deg rotation per tooth per stage
- Quaternions must be unit-normalized
- Move incisors before molars for better staging score

## Real Data Sources
`GET /datasets` lists available clinical datasets. Use `source` param in reset to load real patient anatomy.
