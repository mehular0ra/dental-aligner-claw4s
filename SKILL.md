---
name: dental-aligner-trajectory-planner
description: >
  24-step sequential RL environment for orthodontic aligner trajectory 
  planning in SE(3). Features tool-use actions, Andrews' Six Keys occlusion 
  scoring, PDL biomechanical model, adversarial patient non-compliance, 
  8-axis adaptive difficulty, and real clinical data from Open-Full-Jaw.
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

An RL environment where an AI agent plans orthodontic aligner treatment one stage at a time (24 sequential decisions), moving 28 teeth from malocclusion to alignment in SE(3) space with clinically grounded scoring.

## Step 1: Install dependencies

```bash
uv sync
```

## Step 2: Start the environment server

```bash
uv run python -m server.app &
sleep 5
```

## Step 3: Health check + discover capabilities

```bash
curl -s http://localhost:7860/health
curl -s http://localhost:7860/constraints | python3 -m json.tool
```

Expected: 28 teeth, 24 stages, 0.25mm/2.0deg per-stage limits.

## Step 4: Explore available datasets, difficulty axes, and clinical scoring

```bash
# Real clinical data sources
curl -s http://localhost:7860/datasets | python3 -m json.tool

# 8-axis adaptive difficulty parameters
curl -s http://localhost:7860/difficulty | python3 -c "import sys,json; d=json.load(sys.stdin); print('Axes:', list(d['ranges'].keys()))"

# Andrews' Six Keys occlusion criteria
curl -s http://localhost:7860/occlusion_criteria | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {c}') for c in d['criteria']]"

# PDL biomechanical model
curl -s http://localhost:7860/biomechanics | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Model: {d[\"model\"]}'); print(f'Stiffness: {d[\"stiffness_n_per_mm\"]}')"

# Malocclusion classification patterns
curl -s http://localhost:7860/malocclusion_classes | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {k}: {v[\"description\"]}') for k,v in d['malocclusion_classes'].items()]"

# Adversarial non-compliance types
curl -s http://localhost:7860/noncompliance_types | python3 -m json.tool
```

## Step 5: Reset a stepwise episode

```bash
curl -s -X POST http://localhost:7860/reset_stepwise \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_easy","seed":42,"source":"synthetic","episode_id":"demo_1"}'
```

Returns observation with `current_config` (28x7 tooth poses), `target_config`, `current_stage=0`, `stages_remaining=24`.

## Step 6: Use tools before committing

```bash
# Inspect a specific tooth
curl -s -X POST http://localhost:7860/tool \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"demo_1","tool":"inspect_tooth","args":{"tooth_id":11}}'

# Check for inter-tooth collisions
curl -s -X POST http://localhost:7860/tool \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"demo_1","tool":"check_collisions","args":{}}'
```

## Step 7: Run a full 24-step episode with clinical scoring

```bash
python3 -c "
import json, math, urllib.request

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req).read().decode(), strict=False)

# Reset with adaptive difficulty
obs = post('http://localhost:7860/reset_stepwise', {
    'task_id':'task_easy', 'seed':42, 'source':'synthetic', 'episode_id':'demo_full',
    'difficulty_params': {'n_perturbed_teeth': 12, 'translation_magnitude': 4.0, 'jitter_probability': 0.1}
})
init, tgt = obs['current_config'], obs['target_config']

# SLERP baseline: 24 sequential commits
for stage in range(1, 25):
    alpha = stage / 25.0
    poses = []
    for i in range(28):
        q = [init[i][j]*(1-alpha) + tgt[i][j]*alpha for j in range(4)]
        qn = math.sqrt(sum(x*x for x in q))
        q = [x/qn for x in q]
        t = [init[i][4+j]*(1-alpha) + tgt[i][4+j]*alpha for j in range(3)]
        poses.append(q + t)
    o = post('http://localhost:7860/step_stepwise', {'episode_id':'demo_full', 'poses':poses})
    bd = o.get('reward_breakdown', {})
    evt = bd.get('noncompliance_event')
    if stage % 6 == 0 or stage == 24 or evt:
        prefix = f'[{evt[\"type\"]}] ' if evt else ''
        print(f'Stage {stage:2d}: {prefix}reward={bd.get(\"step_reward\",0):.4f}  occ={bd.get(\"occlusion_composite\",0):.3f}  pdl={bd.get(\"pdl_feasibility\",0):.2f}  collision={bd.get(\"collision_free\",0):.3f}')

print(f'\\nTerminal reward: {o[\"terminal_reward\"]:.4f}')
print(f'Done: {o[\"done\"]}')
"
```

## Step 8: Verify GRPO training readiness

```bash
# Test the GRPO training pipeline (generates prompts, validates reward functions)
uv run python train_grpo.py --test --episodes 3
```

Expected: prompts generated from the environment, reward functions validated, ready for GPU training with `--model Qwen/Qwen2.5-0.5B-Instruct`.

## Step 9: Validate environment features

```bash
# List all API endpoints
curl -s http://localhost:7860/tasks | python3 -c "import sys,json; print(json.load(sys.stdin))"
curl -s http://localhost:7860/health
```

## Environment Design Summary

**Episode:** 24 sequential decisions. Agent observes tooth poses, commits one stage at a time.

**State/Action:** 28x7 arrays — one SE(3) pose per tooth: `[qw,qx,qy,qz,tx,ty,tz]`.

**Tools:** `inspect_tooth`, `simulate_step`, `check_collisions`, `commit_stage`, `rollback_stage`.

**Dense reward:** progress (40%) + compliance (30%) + smoothness (20%) + staging (10%). Each step also reports occlusion composite (Andrews' Six Keys), PDL biomechanical feasibility, and collision-free score.

**Occlusion scoring (Andrews' Six Keys + ABO):** 9 metrics — molar relationship, overjet, overbite, crown angulation, crown inclination, rotations, contact tightness, curve of Spee, arch symmetry.

**Biomechanical PDL model:** Per-tooth-type spring stiffness (incisors 0.2 N/mm, molars 0.8 N/mm). Safe force limits from FEA literature (E_PDL = 68.9 MPa).

**Collision detection:** Oriented bounding ellipsoid model with anatomically correct crown dimensions.

**Adversarial non-compliance:** 3 event types (missed wear, broken attachment, partial wear) triggered stochastically. Simulates real patient non-compliance.

**Adaptive difficulty:** 8 continuous axes with curriculum controller. Auto-escalation at >0.8 for 3 consecutive episodes.

**Malocclusion classes:** 8 clinically classified patterns (Angle Class I/II/III, open bite, crossbite, crowding, spacing, asymmetric).

**Arch forms:** 5 parametric curves (ovoid, tapered, square, catenary, beta function).

**Real clinical data:** Open-Full-Jaw (17 patients), Teeth3DS+ (1,800 scans), Mendeley Jaw (pre-segmented STLs).

**Domain:** Orthodontic aligner treatment planning ($4B+ industry). SE(3) trajectory planning with non-commutative rotations, biomechanical constraints, and clinical scoring standards.
