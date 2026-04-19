# CLAW4S Implementation Spec — battisiBot v2

> Deadline: April 20, 2026
> Goal: Transform 1-shot grading API → 24-step sequential RL environment with tool-use + real data

---

## Completed

- [x] **1. Create new repo** — `dental-aligner-claw4s` from existing env
- [x] **2. Dataset loader** — `server/dataset_loader.py`
  - [x] 2a. `rotation_matrix_to_quaternion()` — Shepperd's method
  - [x] 2b. `_pca_rotation_matrix()` — PCA-based rotation from vertices
  - [x] 2c. `_read_stl_vertices()` — binary STL reader (no trimesh dep)
  - [x] 2d. `load_open_full_jaw()` — JSON teeth axes → (28,7) config
  - [x] 2e. `load_teeth3ds()` — OBJ + JSON labels → (28,7) config
  - [x] 2f. `load_mendeley_jaw()` — individual STL files → (28,7) config
  - [x] 2g. Dataset registry (`DATASET_SOURCES`, `list_datasets()`)
- [x] **3. Integrate dataset loader** — `server/synthetic_data.py`
  - [x] 3a. `generate_case_from_dataset(source, patient_path, difficulty, seed)`
  - [x] 3b. Fill missing teeth with ideal positions
  - [x] 3c. Apply synthetic malocclusion to real target
  - [x] 3d. Generate baseline SLERP trajectory
- [x] **4. Download + validate real data**
  - [x] 4a. Clone Open-Full-Jaw repo (Git LFS)
  - [x] 4b. Extract Patient_1 and inspect JSON structure
  - [x] 4c. Test loader: 13 teeth loaded, quaternions unit-normalized
  - [x] 4d. Test full pipeline: real data → malocclusion → trajectory → PASSED
  - [ ] 4e. Pull all 17 patients (LFS pull running in background)

---

## In Progress

### 5. Stepwise models — `models.py`

- [x] 5a. `StepwiseObservation` Pydantic model
  - Fields: `current_stage: int`, `stages_remaining: int`, `current_config: list[list[float]]` (28x7), `target_config: list[list[float]]` (28x7), `per_tooth_progress: list[float]` (28 values, % done), `cumulative_violations: int`, `step_reward: float`, `stage_history_summary: str`, `task_id: str`, `done: bool`
- [x] 5b. `StepwiseAction` Pydantic model
  - Fields: `poses: list[list[float]]` (28x7 — tooth poses for next stage)
- [x] 5c. `ToolCall` Pydantic model
  - Fields: `tool: str` (one of: inspect_tooth, simulate_step, check_collisions, commit_stage, rollback_stage), `args: dict`
- [x] 5d. `ToolResult` Pydantic model
  - Fields: `tool: str`, `result: dict`, `success: bool`

### 6. Stepwise environment — `server/dental_environment.py`

- [x] 6a. `StepwiseDentalEnvironment` class skeleton
  - [x] `__init__()` — session storage, grader ref
  - [x] `reset(task_id, seed, source, patient_path, difficulty_params)` → StepwiseObservation
  - [x] `step(poses)` → StepwiseObservation (advance one stage)
  - [x] `handle_tool(tool_call)` → ToolResult
  - [x] `_build_observation()` → StepwiseObservation helper
- [x] 6b. Reset logic
  - [x] Generate case (synthetic or dataset-backed)
  - [x] Initialize trajectory buffer: `(26, 28, 7)` with stage 0 = initial, stage 25 = target
  - [x] Set `current_stage = 0`, `done = False`
  - [x] Return initial observation
- [x] 6c. Step logic (the core loop)
  - [x] Validate input poses shape (28x7)
  - [x] Normalize quaternions
  - [x] Write poses into trajectory buffer at `current_stage + 1`
  - [x] Compute per-step dense reward (see 7.)
  - [x] Increment `current_stage`
  - [x] If `current_stage == 24`: compute terminal reward via grader, set `done = True`
  - [x] Return observation with reward
- [x] 6d. Tool handlers
  - [x] `inspect_tooth(tooth_id)` — return pose, distance to target, constraint budget, neighbor info
  - [x] `simulate_step(poses)` — compute reward preview without advancing stage
  - [x] `check_collisions()` — find tooth pairs closer than threshold
  - [x] `commit_stage(poses)` — same as step() (advances stage)
  - [x] `rollback_stage()` — undo last committed stage (max 2 per episode)
- [x] 6e. Session management
  - [x] `_STEPWISE_SESSIONS` dict keyed by `episode_id`
  - [x] Store: case, trajectory_buffer, current_stage, rollback_count, tool_call_count, reward_history

### 7. Per-step dense reward — `server/dental_environment.py` (inline)

- [x] 7a. `_compute_step_reward()` method in StepwiseDentalEnvironment
  - [x] **Progress delta** (weight 0.4): `sum(dist_before - dist_after) / sum(dist_before)`
  - [x] **Constraint compliance** (weight 0.3): 1.0 minus (violations / max_violations)
  - [x] **Smoothness** (weight 0.2): delta variance between consecutive stages
  - [x] **Staging signal** (weight 0.1): incisors-before-molars in early stages
- [x] 7b. Returns: `{step_reward, progress, compliance, smoothness, staging, violations_this_step}`

**Tested:** Full 24-step episode with real data (Open-Full-Jaw Patient_1). Terminal reward = 0.8693. All tools work (inspect, simulate, commit, rollback, collisions).

### 8. Adaptive difficulty — `server/dental_constants.py` + `server/synthetic_data.py`

- [ ] 8a. Add `ADAPTIVE_DIFFICULTY_RANGES` to `dental_constants.py` (SKIPPING for deadline — fixed 3 levels sufficient for submission)
  - Axes: n_perturbed_teeth (4-28), translation_magnitude (0.5-8.0mm), rotation_magnitude (5-35°), constraint_tightness (0.5-2.0), jitter_probability (0.0-1.0), jitter_magnitude (0.0-0.5)
- [ ] 8b. Add `generate_case_adaptive(params, seed)` to `DentalCaseGenerator`
  - Accept continuous param dict instead of string difficulty
  - Use params to control malocclusion severity
- [ ] 8c. Add `next_difficulty(history)` curriculum controller
  - Auto-escalate when agent scores >0.8 for 3 consecutive episodes

### 9. New API endpoints — `server/app.py`

- [x] 9a. `POST /reset_stepwise` — accepts: task_id, seed, source, patient_path, difficulty_params
- [x] 9b. `POST /step_stepwise` — accepts: episode_id, poses (28x7)
- [x] 9c. `POST /tool` — accepts: episode_id, tool, args
- [x] 9d. `GET /datasets` — returns DATASET_SOURCES registry
- [x] 9e. Existing `/reset`, `/step`, `/demo_run` unchanged (backward compat)

### 10. Submission artifacts

- [x] 10a. `SKILL.md` — OpenClaw-compatible executable skill
  - [ ] YAML metadata header (name, description, version, requires, allowed-tools)
  - [ ] Step 1: Install deps (`uv sync`)
  - [ ] Step 2: Start server
  - [ ] Step 3: Health check
  - [ ] Step 4: List datasets (`curl /datasets`)
  - [ ] Step 5: Reset stepwise with real data (`curl /reset_stepwise`)
  - [ ] Step 6: Inspect a tooth (`curl /tool`)
  - [ ] Step 7: Simulate a step (`curl /tool`)
  - [ ] Step 8: Commit stage 1 (`curl /step_stepwise`)
  - [ ] Step 9: Run 3 more stages showing reward progression
  - [ ] Step 10: Validate final score
- [ ] 10b. `paper/` directory with research note (1-4 pages)
  - [ ] Title + abstract
  - [ ] Motivation: $4B industry, SE(3) complexity, why RL
  - [ ] Sequential episode design: diagram + per-step reward formula
  - [ ] Tool-use action space: table of 5 tools
  - [ ] Real clinical data integration: Open-Full-Jaw pipeline
  - [ ] Adaptive difficulty: continuous parameter axes
  - [ ] Results: comparison table vs hackathon winners
  - [ ] Citations: all datasets, TADPM, Wang et al.
- [x] 10c. `CLAUDE.md` — AI agent quick reference
- [ ] 10d. Update `README.md` — document stepwise mode + tools + datasets

### 11. Verification

- [x] 11a. Original `/reset` + `/step` still work
- [x] 11b. Stepwise mode: 24-step episode completes end-to-end (terminal=0.8693)
- [x] 11c. Real data source: `reset_stepwise` with `source=open_full_jaw` works
- [x] 11d. Tool-use: inspect, simulate, check_collisions, commit, rollback all work
- [ ] 11e. Adaptive difficulty: SKIPPED (3 fixed levels sufficient)
- [ ] 11f. SKILL.md cold-starts from clean clone
- [ ] 11g. Commit and push to new repo
