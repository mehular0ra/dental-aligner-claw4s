---
name: dental-aligner-trajectory-planner
description: >
  24-step sequential RL environment for orthodontic aligner trajectory 
  planning in SE(3). Agents plan one aligner stage at a time across 28 
  teeth with tool-use actions (inspect, simulate, commit, rollback) and 
  real clinical data from Open-Full-Jaw. Includes battisiBot hybrid 
  LLM+SLERP baseline agent.
version: 2.0.0
metadata:
  openclaw:
    requires:
      bins: [uv, curl]
    allowed-tools: Bash(uv *), Bash(curl *), Bash(python3 *)
    emoji: "🦷"
    homepage: https://huggingface.co/spaces/grimoors/dental-aligner-env
---

# Dental Aligner Trajectory Planner — battisiBot v2

An RL environment where an AI agent plans orthodontic aligner treatment one stage at a time (24 sequential decisions), moving 28 teeth from malocclusion to alignment in SE(3) space.

## Step 1: Install dependencies

```bash
uv sync
```

## Step 2: Start the environment server

```bash
uv run python -m server.app &
sleep 3
```

## Step 3: Health check

```bash
curl -s http://localhost:7860/health
```

Expected: `{"status":"healthy"}`

## Step 4: List available clinical datasets

```bash
curl -s http://localhost:7860/datasets | python3 -m json.tool
```

Expected: JSON listing Open-Full-Jaw (17 patients), Teeth3DS+ (1800 scans), Mendeley Jaw (1 patient).

## Step 5: Reset a 24-step stepwise episode with real clinical data

```bash
EPISODE=$(curl -s -X POST http://localhost:7860/reset_stepwise \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_easy","seed":42,"source":"open_full_jaw","patient_path":"data/Open-Full-Jaw/Patient_1"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_id',''))")
echo "Episode started: task=$EPISODE"
```

The observation includes:
- `current_config`: 28 tooth poses as `[qw,qx,qy,qz,tx,ty,tz]` (real patient anatomy)
- `target_config`: target alignment poses
- `current_stage`: 0 (initial)
- `stages_remaining`: 24

## Step 6: Inspect a specific tooth

```bash
curl -s -X POST http://localhost:7860/tool \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"EPISODE_ID","tool":"inspect_tooth","args":{"tooth_id":31}}' \
  | python3 -m json.tool
```

Returns: current pose, target pose, remaining distance (mm/degrees), staging priority, neighbor teeth.

## Step 7: Simulate a candidate step (preview without committing)

```bash
# Generate SLERP poses for stage 1 and preview the reward
python3 -c "
import json, urllib.request
# ... agent would compute poses here
# simulate_step returns reward preview without advancing the episode
req = urllib.request.Request(
    'http://localhost:7860/tool',
    data=json.dumps({'episode_id':'EPISODE_ID','tool':'simulate_step','args':{'poses':POSES}}).encode(),
    headers={'Content-Type':'application/json'}
)
resp = json.loads(urllib.request.urlopen(req).read())
print(f'Preview reward: {resp[\"result\"][\"preview_reward\"]}')
print(f'Violations: {resp[\"result\"][\"violations\"]}')
"
```

## Step 8: Commit stage 1

```bash
curl -s -X POST http://localhost:7860/step_stepwise \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"EPISODE_ID","poses":POSES_ARRAY}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Stage {d[\"current_stage\"]}: reward={d[\"step_reward\"]}')"
```

The agent receives:
- `step_reward`: dense reward (progress 40% + compliance 30% + smoothness 20% + staging 10%)
- `reward_breakdown`: per-component scores
- Updated `current_config` reflecting the committed poses
- `per_tooth_progress`: per-tooth % completion toward target

## Step 9: Run a full 24-step SLERP baseline episode

```bash
python3 -c "
import json, urllib.request, math

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req).read())

# Reset
obs = post('http://localhost:7860/reset_stepwise', {'task_id':'task_easy','seed':42,'source':'synthetic','episode_id':'demo_slerp'})
initial = obs['current_config']
target = obs['target_config']

# SLERP interpolation for 24 stages
for stage in range(1, 25):
    alpha = stage / 25.0
    poses = []
    for i in range(28):
        q0, q1 = initial[i][:4], target[i][:4]
        t0, t1 = initial[i][4:], target[i][4:]
        # Simple linear interp (proper SLERP in production)
        q = [q0[j]*(1-alpha) + q1[j]*alpha for j in range(4)]
        qn = math.sqrt(sum(x*x for x in q))
        q = [x/qn for x in q]
        t = [t0[j]*(1-alpha) + t1[j]*alpha for j in range(3)]
        poses.append(q + t)
    
    obs = post('http://localhost:7860/step_stepwise', {'episode_id':'demo_slerp','poses':poses})
    if stage % 6 == 0 or stage == 24:
        print(f'Stage {stage:2d}: reward={obs[\"step_reward\"]:.4f}')

print(f'Terminal reward: {obs[\"terminal_reward\"]:.4f}')
print(f'Violations: {obs[\"cumulative_violations\"]}')
print(f'Done: {obs[\"done\"]}')
"
```

## Step 10: Validate the environment

```bash
curl -s http://localhost:7860/constraints | python3 -m json.tool
curl -s http://localhost:7860/tasks | python3 -m json.tool
```

Verify: 28 teeth, 24 stages, 0.25mm/2.0deg per-stage limits.

## Environment Design

**Episode structure:** 24 sequential decisions. At each step the agent observes current tooth poses and commits poses for the next aligner stage.

**Action space:** 28x7 array (one SE(3) pose per tooth: unit quaternion + translation).

**Tool-use actions:** `inspect_tooth`, `simulate_step`, `check_collisions`, `commit_stage`, `rollback_stage`. Only `commit_stage` advances the episode.

**Reward:** Dense per-step (progress 40%, compliance 30%, smoothness 20%, staging 10%) + terminal graded score.

**Real data:** Clinical tooth poses from the Open-Full-Jaw dataset (17 patients, CC BY-NC-SA 4.0).

**Domain:** Orthodontic aligner treatment planning is a $4B+ industry. SE(3) trajectory planning over 28 teeth with clinical constraints is a uniquely challenging RL problem involving non-commutative rotations, long-horizon planning, and biomechanical constraints.
