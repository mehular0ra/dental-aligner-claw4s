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

## 6. Occlusion Scoring (Andrews' Six Keys)

We implement 9 clinically grounded occlusion metrics computable directly from the SE(3) pose representation, based on Andrews' Six Keys to Normal Occlusion [6] and the ABO Objective Grading System [7]:

| Metric | Clinical Basis | Computation from SE(3) |
|--------|---------------|----------------------|
| Molar relationship | Angle Class I/II/III | Anteroposterior offset between opposing first molars |
| Overjet | Incisor AP distance | `t_y(upper_incisor) - t_y(lower_incisor)`, ideal 2-3mm |
| Overbite | Incisor vertical overlap | `t_z(upper_incisor) - t_z(lower_incisor)`, ideal 2-3mm |
| Crown angulation | Mesiodistal tip (Key 2) | Pitch angle from quaternion vs Andrews' ideal per tooth type |
| Crown inclination | Labiolingual torque (Key 3) | Roll angle from quaternion vs ideal torque values |
| Rotations | Long-axis rotation (Key 4) | Yaw angle deviation from ideal orientation |
| Contact tightness | No spacing (Key 5) | Inter-tooth centroid distance minus expected crown widths |
| Curve of Spee | Flat occlusal plane (Key 6) | Max deviation of lower arch from fitted plane, ideal 0-2.5mm |
| Arch symmetry | Left-right mirror | Positional difference between corresponding bilateral teeth |

Weighted composite score with configurable weights (molar 20%, overjet 15%, overbite 10%, angulation 10%, inclination 10%, rotations 10%, contacts 10%, Spee 5%, symmetry 10%).

## 7. Biomechanical PDL Model

We implement a simplified Kelvin-Voigt viscoelastic model for the periodontal ligament [8], with per-tooth-type stiffness coefficients calibrated from FEA literature [1,9]:

| Tooth Type | PDL Stiffness (N/mm) | Max Safe Force (N) | Root Morphology |
|------------|:---:|:---:|-----|
| Central incisor | 0.20 | 0.5 | Single root, small |
| Lateral incisor | 0.25 | 0.5 | Single root, small |
| Canine | 0.35 | 0.75 | Single root, longest |
| Premolar | 0.40-0.45 | 0.75 | 1-2 roots |
| First molar | 0.70 | 1.5 | 3 roots, large surface |
| Second molar | 0.80 | 1.5 | 3 roots, large surface |

Material properties: E_PDL = 68.9 MPa, v = 0.45 (from Open-Full-Jaw FEA validation [1]); aligner PETG: E = 1361 MPa, v = 0.30.

The `score_biomechanical_feasibility()` function evaluates whether per-stage tooth movements stay within safe PDL force limits, penalizing trajectories that risk root resorption.

## 8. Ablation Results

We validate that the environment produces clinically meaningful treatment progression by tracking occlusion metrics across all 24 stages (medium difficulty, synthetic full-arch data, SLERP baseline):

| Stage | Reward | Occlusion | Rotations | Symmetry | Spee | Overbite |
|:-----:|:------:|:---------:|:---------:|:--------:|:----:|:--------:|
| 1 | 0.616 | 0.623 | 0.777 | 0.607 | 0.623 | 0.739 |
| 8 | 0.622 | 0.703 | 0.844 | 0.722 | 0.976 | 0.961 |
| 16 | 0.620 | 0.688 | 0.918 | 0.853 | 1.000 | 0.700 |
| 24 | 0.780 | 0.701 | 0.991 | 0.984 | 1.000 | 1.000 |

Key findings:
- Occlusion composite improves +14.6% (0.612 initial malocclusion → 0.701 post-treatment)
- Rotations normalize: 0.777 → 0.991 (teeth de-rotating toward ideal)
- Arch symmetry: 0.607 → 0.984
- Curve of Spee flattens: 0.623 → 1.000
- PDL feasibility = 1.0 throughout (all movements within safe biomechanical limits)

**Multi-source data validation:** The environment loads real clinical data from 3 independent sources:
- Open-Full-Jaw [1]: 17 patients, JSON principal axes → SE(3) (validated)
- Mendeley Jaw [3]: 1 patient, 11 pre-segmented STL teeth → PCA poses (validated)
- Synthetic: procedural arch with parameterized malocclusion (validated)

## 9. Reproducibility

- Deterministic seeding: same seed → same case (verified across runs)
- Pinned dependencies via `uv.lock`
- Docker support via `Dockerfile`
- All endpoints documented in SKILL.md with exact `curl` commands
- SLERP baseline scores 0.87-0.89 terminal reward
- Cold-start from clean clone verified

## References

[1] A. Thoeni, F. Guenther, M. Tummala, J. Hedegaard, K. Nishimura, F. Bayer, D. Manzoni, and K. Erleben. "Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw." arXiv:2209.07576, 2022.

[2] A. Ben-Hamadou, O. Torresin, F. Michelutti, F. Oleari, and E. Vezzetti. "Teeth3DS+: A benchmark for 3D teeth segmentation and labeling from intra-oral 3D scans." arXiv:2210.06094, 2022.

[3] Universidad Autonoma de Occidente. "Data of synthetic 3D models of the human jaw, including teeth, ligaments, and bone structures." Mendeley Data, DOI: 10.17632/xjsx7nfhj8.1, 2024.

[4] S. Wang, C. Lei, Y. Liang, J. Sun, X. Xie, Y. Wang, F. Zuo, Y. Bai, S. Li, and Y.-J. Liu. "A 3D dental model dataset with pre/post-orthodontic treatment for automatic tooth alignment." Scientific Data, 11:1277, 2024. DOI: 10.1038/s41597-024-04138-7.

[5] C. Lei, M. Xia, S. Wang, Y. Liang, R. Yi, Y.-H. Wen, and Y.-J. Liu. "Automatic tooth arrangement with joint features of point and mesh representations via diffusion probabilistic models." Computer Aided Geometric Design, 111:102293, 2024. DOI: 10.1016/j.cagd.2024.102293.

[6] L. F. Andrews. "The six keys to normal occlusion." American Journal of Orthodontics, 62(3):296-309, 1972. DOI: 10.1016/S0002-9416(72)90268-0.

[7] American Board of Orthodontics. "ABO Objective Grading System for Dental Casts and Panoramic Radiographs." https://americanboardortho.com.

[8] J. Mena-Mena and C. Giannini. "An efficient spring model for an integrated orthodontic tooth movement." Applied Sciences, 13(8):5013, 2023. DOI: 10.3390/app13085013.

[9] A. Thoeni et al. "A systematic comparison between FEBio and PolyFEM for biomechanical simulation of the human body." In: Computational Biomechanics for Medicine, 2024. PMC: 10843651.

[10] Y. Dou, H. Wu, C. Li, C. Wang, T. Yang, M. Zhu, D. Shen, and Z. Cui. "CLIK-Diffusion: Clinical Knowledge-informed Diffusion Model for Tooth Alignment." Medical Image Analysis, 2025.

[11] G. Wei, Z. Cui, Y. Liu, N. Chen, R. Chen, G. Li, and W. Wang. "TANet: Towards Fully Automatic Tooth Arrangement." In: ECCV, pp. 481-497, 2020.

[12] M. Truong, S. Chang, and K. Vo. "Zero Shot Cancer: Autonomous Biologist POMDP Environment." OpenEnv Hackathon SF, 2nd Place, 2026. GitHub: mhtruong1031/OpenENV-Hackathon.

[13] S. R. Potu, G. Yu, and A. Ranjan. "Kube SRE Gym: Self-improving RL for Kubernetes Incident Response." OpenEnv Hackathon SF, 1st Place, 2026. GitHub: sid-rp/kube-sre-gym.

[14] Multi-task reinforcement learning and explainable AI-driven platform for personalized orthodontic-orthognathic treatment planning. Scientific Reports, 15:09236, 2025. DOI: 10.1038/s41598-025-09236-z.

[15] Borghi et al. "Bits2Bites: Intra-oral Scans Occlusal Classification." ODIN Workshop at MICCAI, 2025.

[16] S. Wang et al. "Computational Orthodontic Force Simulation: A Review." arXiv:2503.24195, 2025.

[17] 3D FEA for Clear Aligner Tooth Movement. Scientific Reports, 14:63907, 2024. DOI: 10.1038/s41598-024-63907-x.
