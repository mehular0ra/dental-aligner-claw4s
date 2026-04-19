# battisiBot: A 24-Step Sequential RL Environment for Orthodontic Aligner Trajectory Planning in SE(3) with Tool-Use Actions and Adaptive Difficulty

**Authors:** Mehul Arora, Vivek Mathur, Claw, Bradly Alicea

**Affiliations:** Orthogonal Research and Education Laboratory; CCNSB, IIIT Hyderabad; University of Illinois Urbana-Champaign

---

## Abstract

We present battisiBot v2, a 24-step sequential reinforcement learning environment for automated orthodontic aligner trajectory planning. An AI agent plans one aligner stage at a time across 28 teeth represented as SE(3) poses (unit quaternion + translation), receiving dense per-step reward and access to 5 tool-use actions (inspect, simulate, commit, rollback, check collisions). The environment supports real clinical data from the Open-Full-Jaw dataset (17 patients) alongside synthetic generation with 8-axis adaptive difficulty curriculum. This transforms orthodontic treatment planning from a single-shot optimization into a sequential decision-making problem suitable for RL training with GRPO/DAPO.

---

## 1. Motivation

Orthodontic treatment with clear aligners (e.g., Invisalign) requires dividing corrective tooth movement into a sequence of small steps, each corresponding to one physical aligner tray. This is a $4B+ industry where manual planning takes 2-3 hours per case.

The problem is non-trivial for RL because:
- **SE(3) geometry**: Each of 28 teeth occupies a pose with 3 translational + 3 rotational degrees of freedom. Rotations are non-commutative; naive Euler angle interpolation produces gimbal lock.
- **Long-horizon planning**: 24 sequential decisions with compounding effects across a (28 x 6)-dimensional configuration space.
- **Clinical constraints**: Per-stage movement limits (0.25mm translation, 2.0deg rotation per tooth), staging priority (incisors before molars), and collision avoidance.

No existing RL benchmark operates in SE(3) with these clinical constraints.

## 2. Environment Design

### 2.1 Episode Structure

Each episode consists of 24 sequential steps. At step *t*, the agent observes the current 28-tooth configuration and commits poses for the next aligner stage:

```
reset() -> observe stage 0
  -> step(stage_1_poses) -> reward_1 + observe stage 1
  -> step(stage_2_poses) -> reward_2 + observe stage 2
  -> ... (24 steps)
  -> step(stage_24_poses) -> terminal_reward + done
```

### 2.2 State and Action Space

**State**: 28 x 7 array -- one SE(3) pose per tooth as `[qw, qx, qy, qz, tx, ty, tz]` where `(qw, qx, qy, qz)` is a unit quaternion and `(tx, ty, tz)` is translation in mm.

**Action**: 28 x 7 array -- proposed poses for the next stage. Quaternions are auto-normalized.

### 2.3 Tool-Use Actions

Beyond raw pose submission, agents can use 5 tools that enable emergent planning strategies:

| Tool | Effect | Advances Episode? |
|------|--------|:-:|
| `inspect_tooth(id)` | Returns pose, distance to target, staging priority, neighbors | No |
| `simulate_step(poses)` | Preview reward without committing | No |
| `check_collisions()` | Detect tooth pairs closer than threshold | No |
| `commit_stage(poses)` | Finalize stage (same as step) | Yes |
| `rollback_stage()` | Undo last commit (max 2 per episode) | Reverses |

This produces emergent strategies: inspect -> simulate multiple options -> commit the best, or commit speculatively and rollback if the reward is poor.

### 2.4 Dense Per-Step Reward

Each step returns a weighted reward:

| Component | Weight | Description |
|-----------|:------:|-------------|
| Progress | 0.4 | Fraction of remaining distance closed this step |
| Compliance | 0.3 | 1.0 minus (violations / max_possible_violations) |
| Smoothness | 0.2 | Low variance in movement magnitudes vs previous step |
| Staging | 0.1 | Bonus if incisors moved before molars at this stage |

Terminal reward uses the full 5-component grader (accuracy, smoothness, compliance, staging quality, recovery bonus).

## 3. Real Clinical Data Integration

We integrate the Open-Full-Jaw dataset [1] (17 patients, CC BY-NC-SA 4.0) which provides per-tooth principal axes as JSON:

```json
{"18": {"c": [90.97, 73.65, 38.17], "x": [...], "y": [...], "z": [...]}}
```

Our `dataset_loader.py` converts these to SE(3) poses:
1. Center of mass -> translation `[tx, ty, tz]`
2. Axis endpoints -> rotation matrix (orthogonalized via SVD) -> unit quaternion `[qw, qx, qy, qz]`

The loader also supports Teeth3DS+ [2] (1,800 scans, per-vertex FDI labels -> PCA-based poses) and Mendeley Jaw Models [3] (pre-segmented STLs).

Real anatomy serves as target configurations; synthetic malocclusion is applied to create initial states. This ensures the RL agent trains on clinically grounded geometry rather than purely procedural arches.

## 4. Adaptive Difficulty

We replace fixed difficulty levels with an 8-axis continuous curriculum:

| Axis | Range | Description |
|------|-------|-------------|
| n_perturbed_teeth | 4-28 | Number of displaced teeth |
| translation_magnitude | 0.5-8.0 mm | Max displacement |
| rotation_magnitude | 5-35 deg | Max rotation |
| multi_axis_rotation | bool | Single vs multi-axis |
| constraint_tightness | 0.5-2.0 | Budget multiplier |
| jitter_probability | 0.0-1.0 | Adversarial perturbation chance |
| jitter_magnitude | 0.0-0.5 | Perturbation strength |
| missing_teeth | 0-4 | Teeth requiring no movement |

A `CurriculumController` tracks per-episode rewards and auto-escalates the weakest axis when the agent scores >0.8 for 3 consecutive episodes.

## 5. Comparison with Hackathon Winners

We compare our environment design against the top submissions from the OpenEnv Hackathon SF (March 2026):

| Dimension | Kube SRE Gym (1st) | Zero Shot Cancer (2nd) | ShopRLVE (3rd) | battisiBot v2 |
|-----------|:--:|:--:|:--:|:--:|
| Multi-step episodes | 5-30 steps | 21 tool calls | 8 tasks | **24 steps** |
| Tool-use actions | kubectl | 40+ bio tools | 12 verifiers | **5 tools** |
| Reward components | LLM judge + phases | 7 components | algorithmic | **4 dense + 5 terminal** |
| Adaptive difficulty | adversarial designer | domain randomization | 12-axis curriculum | **8-axis curriculum** |
| Domain uniqueness | K8s ops | Biology POMDP | E-commerce | **SE(3) dental** |
| Math depth | Moderate | Moderate | Low | **Quaternion SLERP, Lie groups** |
| Real data | Live GKE cluster | DOI-linked | Product catalogs | **Clinical CT scans** |

## 6. Reproducibility

- Deterministic seeding: same seed -> same case
- Pinned dependencies via `uv.lock`
- Docker support via `Dockerfile`
- All endpoints documented in SKILL.md with exact `curl` commands
- SLERP baseline scores ~0.87 terminal reward; a random agent scores ~0.15

## References

[1] Open-Full-Jaw: A. Thoeni et al. "Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw." arXiv:2209.07576, 2022.

[2] Teeth3DS+: A. Ben-Hamadou et al. "Teeth3DS+: A benchmark for 3D teeth segmentation and labeling from intra-oral 3D scans." arXiv:2210.06094, 2022.

[3] Mendeley Jaw Models: Universidad Autonoma de Occidente. DOI: 10.17632/xjsx7nfhj8.1, 2024.

[4] Wang et al. "A 3D dental model dataset with pre/post-orthodontic treatment for automatic tooth alignment." Scientific Data, 11:1277, 2024.

[5] Lei et al. "Automatic tooth arrangement with joint features of point and mesh representations via diffusion probabilistic models." CAGD, 111:102293, 2024.
