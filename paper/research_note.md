# battisiBot: A 24-Step Sequential RL Environment for Orthodontic Aligner Trajectory Planning in SE(3)

**Authors:** Mehul Arora, Vivek Mathur, Claw, Bradly Alicea

**Affiliations:** Orthogonal Research and Education Laboratory; CCNSB, IIIT Hyderabad; University of Illinois Urbana-Champaign

---

## Abstract

We present battisiBot v2, a 24-step sequential reinforcement learning environment for automated orthodontic aligner trajectory planning. An AI agent plans one aligner stage at a time across 28 teeth represented as SE(3) poses (unit quaternion + translation), receiving dense per-step reward and access to 5 tool-use actions. The environment features Andrews' Six Keys occlusion scoring [1], a Kelvin-Voigt PDL biomechanical model [2], oriented ellipsoid collision detection, adversarial patient non-compliance simulation, 8-axis adaptive difficulty with curriculum control [3], 8 clinically classified malocclusion patterns, 5 parametric arch forms, and real clinical data from 3 independent sources [4,5,6]. This transforms orthodontic treatment planning from a single-shot optimization into a sequential decision-making problem suitable for RL training with GRPO [7] or DAPO [8].

---

## 1. Motivation

Orthodontic treatment with clear aligners is a $4B+ industry where manual planning takes 2-3 hours per case. Recent work on automated tooth arrangement using deep learning [9,10,11,12,13] has demonstrated the feasibility of predicting post-treatment configurations, but none formulate the problem as a sequential RL task with per-step feedback.

The problem is non-trivial for RL because: (i) each of 28 teeth occupies a pose in SE(3) with 6 degrees of freedom — rotations are non-commutative, requiring quaternion SLERP [14] or Lie-group geodesics [15] rather than naive Euler interpolation; (ii) a 24-stage plan defines a path through a (28 x 6)-dimensional configuration space with compounding effects; (iii) clinical constraints impose hard limits on per-stage movement (0.25mm translation, 2.0deg rotation) with staging priority requirements [1].

The closest existing work is the multi-task RL platform for orthodontic-orthognathic treatment [16], which uses Soft Actor-Critic on 347 cases but operates at the treatment decision level rather than tooth-by-tooth trajectory planning, and does not provide an open RL environment.

## 2. Environment Design

### 2.1 Episode Structure

Each episode consists of 24 sequential steps. At step *t*, the agent observes the current 28-tooth configuration and commits poses for the next aligner stage, receiving dense reward feedback.

### 2.2 State and Action Space

**State**: 28 x 7 array — one SE(3) pose per tooth as `[qw, qx, qy, qz, tx, ty, tz]` following the scalar-first quaternion convention [14].

**Action**: 28 x 7 array — proposed poses for the next stage. Quaternions are auto-normalized.

### 2.3 Tool-Use Actions

Following the ToolRL paradigm [17], agents access 5 tools: `inspect_tooth` (pose/distance/priority), `simulate_step` (reward preview without commit), `check_collisions` (inter-tooth penetration), `commit_stage` (finalize), and `rollback_stage` (undo, max 2 per episode). This enables emergent planning strategies analogous to the tool-use patterns observed in recent agentic RL work [18,19].

### 2.4 Dense Per-Step Reward

Each step returns a weighted reward: progress toward target (40%), constraint compliance (30%), smoothness (20%), and staging signal (10%). Terminal reward uses a 5-component grader aligned with clinical evaluation standards.

## 3. Clinical Grounding

### 3.1 Occlusion Scoring (Andrews' Six Keys)

We implement 9 metrics computable directly from SE(3) poses, based on Andrews' foundational work [1] and the ABO Objective Grading System [20]: molar relationship (Angle classification), overjet (ideal 2-3mm), overbite (ideal 2-3mm), crown angulation (Key 2), crown inclination (Key 3), long-axis rotations (Key 4), contact tightness (Key 5), curve of Spee (Key 6) [21], and arch symmetry. Bolton analysis [22,23] informs expected tooth size ratios.

### 3.2 Biomechanical PDL Model

We model the periodontal ligament as a Kelvin-Voigt viscoelastic element [2,24] with per-tooth-type stiffness calibrated from FEA literature [4,25]: incisors 0.2 N/mm, canines 0.35 N/mm, molars 0.7-0.8 N/mm (reflecting 3-5x greater root surface area [26]). Material properties (E_PDL = 68.9 MPa, v = 0.45) are from the Open-Full-Jaw FEM validation [4,25]. Safe force limits prevent trajectories that risk root resorption [27].

### 3.3 Collision Detection

Oriented bounding ellipsoids with anatomically accurate crown dimensions (from Wheeler's Dental Anatomy) approximate each tooth. Surface distance is computed via the GJK support function on ellipsoids, with a configurable safety margin (default 0.3mm).

### 3.4 Adversarial Patient Non-Compliance

Three clinically grounded event types simulate real non-compliance [28]: (i) *missed wear* — teeth partially revert toward pre-treatment positions (20-50% reversal); (ii) *broken attachment* — single tooth resets to initial pose; (iii) *partial wear* — all movements at 40-60% efficacy. Events trigger stochastically with probability controlled by the curriculum.

## 4. Domain Randomization

### 4.1 Malocclusion Classification

We replace random perturbations with 8 clinically classified patterns following the Angle classification [29]: Class I crowding, Class I spacing, Class II division 1 (upper protrusion), Class II division 2 (retroclined incisors + deep bite), Class III (mandibular protrusion), open bite, posterior crossbite, and asymmetric malocclusion. Each pattern applies anatomically correct perturbation logic (e.g., Class II div 1 proclination of upper incisors + increased overjet).

### 4.2 Parametric Arch Forms

Five arch shapes replace the hardcoded parabolic arch [30,31]: ovoid (standard), tapered (V-shaped), square (U-shaped), catenary (y = a*cosh(x/a)), and beta function (Y = AX^6 + BX^2). Parameters are randomized per episode for domain diversity.

### 4.3 Adaptive Difficulty

An 8-axis continuous curriculum replaces fixed difficulty levels [3]: n_perturbed_teeth (4-28), translation_magnitude (0.5-8.0mm), rotation_magnitude (5-35deg), multi_axis_rotation, constraint_tightness (0.5-2.0x), jitter_probability (0.0-1.0), jitter_magnitude (0.0-0.5), and missing_teeth (0-4). A `CurriculumController` auto-escalates when the agent scores >0.8 for 3 consecutive episodes.

## 5. Real Clinical Data

We integrate 3 independent data sources, each converted to the standard (28, 7) SE(3) format:

- **Open-Full-Jaw** [4]: 17 patients with per-tooth principal axes (JSON) from CBCT segmentation. Axes are orthogonalized via SVD and converted to unit quaternions.
- **Teeth3DS+** [5]: 1,800 intraoral scans with per-vertex FDI segmentation labels. Per-tooth centroid and PCA-based orientation extraction. Also available via PyTorch Geometric [32].
- **Mendeley Jaw Models** [6]: Pre-segmented individual tooth STL files (14 lower teeth + PDL meshes). Binary STL reader extracts vertices for centroid + PCA computation.

Additional datasets pending access include the Tsinghua 3D Orthodontic Dataset [33] (1,060 pre/post-treatment pairs), CLIK-Diffusion [10] (448 paired models with 4x4 transformation matrices), and Bits2Bites [34] (200 paired arches with multi-label occlusion classification).

## 6. Ablation Results

We track occlusion metrics across 24 stages (medium difficulty, SLERP baseline):

| Stage | Reward | Occlusion | Rotations | Symmetry | Curve of Spee | Overbite |
|:-----:|:------:|:---------:|:---------:|:--------:|:---:|:---:|
| 1 | 0.616 | 0.623 | 0.777 | 0.607 | 0.623 | 0.739 |
| 8 | 0.622 | 0.703 | 0.844 | 0.722 | 0.976 | 0.961 |
| 16 | 0.620 | 0.688 | 0.918 | 0.853 | 1.000 | 0.700 |
| 24 | 0.780 | 0.701 | 0.991 | 0.984 | 1.000 | 1.000 |

Occlusion composite improves +14.6% (0.612 → 0.701). PDL feasibility = 1.0 throughout. With adversarial non-compliance (jitter_probability=0.5), terminal reward decreases from 0.888 to 0.878, confirming events have measurable impact on treatment quality.

## 7. Reproducibility

Deterministic seeding ensures identical cases across runs. Dependencies are pinned via `uv.lock`. All endpoints are documented in SKILL.md with exact `curl` commands. SLERP baseline scores 0.87-0.89 terminal reward. Cold-start from clean clone verified.

---

## References

### Orthodontic Clinical Standards
[1] L.F. Andrews. "The six keys to normal occlusion." *American Journal of Orthodontics*, 62(3):296-309, 1972. DOI: 10.1016/S0002-9416(72)90268-0.

[20] J.S. Casko, J.L. Vaden, V.G. Kokich, et al. "Objective grading system for dental casts and panoramic radiographs." *AJODO*, 114(5):589-599, 1998. DOI: 10.1016/S0889-5406(98)70179-9.

[21] P. Chitra and M. Balasubramaniam. "Significance of curve of Spee: An orthodontic review." *J. Pharm. Bioallied Sci.*, 4(Suppl 2):S313-S316, 2012. DOI: 10.4103/0975-7406.100266.

[22] W.A. Bolton. "Disharmony in tooth size and its relation to the analysis and treatment of malocclusion." *Angle Orthodontist*, 28(3):113-130, 1958.

[23] W.A. Bolton. "The clinical application of a tooth-size analysis." *Am. J. Orthod.*, 48(7):504-529, 1962. DOI: 10.1016/0002-9416(62)90129-X.

[28] G. Al-Jamal et al. "Clear aligner compliance: A systematic review." *Angle Orthodontist*, 93(5):593-601, 2023. DOI: 10.2319/010423-6.1.

[29] E.H. Angle. "Classification of malocclusion." *Dental Cosmos*, 41:248-264, 1899.

### Biomechanics and PDL Modeling
[2] S. Yona, O. Medina, R. Sarig, and N. Shvalb. "An efficient spring model for an integrated orthodontic tooth movement." *Applied Sciences*, 13(8):5013, 2023. DOI: 10.3390/app13085013.

[24] S.R. Toms et al. "Viscoelasticity of periodontal ligament: An analytical model." *Mechanics of Advanced Materials and Modern Processes*, 1:7, 2015. DOI: 10.1186/s40759-015-0007-0.

[25] T. Gholamalizadeh et al. "A systematic comparison between FEBio and PolyFEM for biomechanical simulation." PMC: 10843651, 2024.

[26] T.S. Fill et al. "The biomechanical function of periodontal ligament fibres in orthodontic tooth movement." *PLoS ONE*, 9(7):e102387, 2014. DOI: 10.1371/journal.pone.0102387.

[27] S. Wang et al. "Computational orthodontic force simulation: A review." arXiv: 2503.24195, 2025.

### Dental AI and Tooth Arrangement
[9] G. Wei et al. "TANet: Towards fully automatic tooth arrangement." *ECCV 2020*, pp. 481-497. DOI: 10.1007/978-3-030-58555-6_29.

[10] Y. Dou et al. "CLIK-Diffusion: Clinical knowledge-informed diffusion model for tooth alignment." *Medical Image Analysis*, 2025. DOI: 10.1016/j.media.2025.103452.

[11] C. Lei et al. "Automatic tooth arrangement with joint features of point and mesh representations via diffusion probabilistic models." *CAGD*, 111:102293, 2024. DOI: 10.1016/j.cagd.2024.102293.

[12] Z. Dong and J. Chen. "Transformer-based tooth alignment prediction with occlusion and collision constraints." *ICCV 2025*. arXiv: 2410.20806.

[13] L. Yang et al. "iOrthoPredictor: Model-guided deep prediction of teeth alignment." *ACM TOG (SIGGRAPH Asia)*, 39(6):220, 2020. DOI: 10.1145/3414685.3417771.

[16] Z. Li and L. Wang. "Multi-task reinforcement learning for personalized orthodontic-orthognathic treatment planning." *Scientific Reports*, 15:24502, 2025. DOI: 10.1038/s41598-025-09236-z.

### SE(3) and Quaternion Methods
[14] K. Shoemake. "Animating rotation with quaternion curves." *SIGGRAPH '85*, pp. 245-254. DOI: 10.1145/325334.325242.

[15] F. Fuchs et al. "SE(3)-Transformers: 3D roto-translation equivariant attention networks." *NeurIPS*, 33:1970-1981, 2020. arXiv: 2006.10503.

### RL Algorithms and Environments
[3] S. Narvekar et al. "Curriculum learning for reinforcement learning domains: A framework and survey." *JMLR*, 21(181):1-50, 2020. arXiv: 2003.04960.

[7] Z. Shao et al. "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." arXiv: 2402.03300, 2024.

[8] "DAPO: An open-source LLM reinforcement learning system at scale." arXiv: 2503.14476, 2025.

[17] C. Qian et al. "ToolRL: Reward is all tool learning needs." *NeurIPS 2025*. arXiv: 2504.13958.

[18] "SkillRL: Evolving agents via recursive skill-augmented reinforcement learning." arXiv: 2602.08234, 2026.

[19] "SKILL0: In-context agentic reinforcement learning for skill internalization." arXiv: 2604.02268, 2026.

### Datasets
[4] T. Gholamalizadeh et al. "Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw." *CMPB*, 224:107009, 2022. DOI: 10.1016/j.cmpb.2022.107009.

[5] A. Ben-Hamadou et al. "Teeth3DS+: A benchmark for 3D teeth segmentation and labeling from intra-oral 3D scans." arXiv: 2210.06094, 2022.

[6] Universidad Autonoma de Occidente. "Synthetic 3D models of the human jaw." *Mendeley Data*, DOI: 10.17632/xjsx7nfhj8.1, 2024.

[32] M. Fey and J.E. Lenssen. "Fast graph representation learning with PyTorch Geometric." *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019. arXiv: 1903.02428.

[33] S. Wang et al. "A 3D dental model dataset with pre/post-orthodontic treatment for automatic tooth alignment." *Scientific Data*, 11:1277, 2024. DOI: 10.1038/s41597-024-04138-7.

[34] G. Borghi et al. "Bits2Bites: Intra-oral scans occlusal classification." *ODIN Workshop, MICCAI 2025*.

### Arch Form and Domain Randomization
[30] S. Braun et al. "Form of the human dental arch." *Angle Orthodontist*, 68(1):29-36, 1998.

[31] S. AlHarbi et al. "Mathematical analyses of dental arch curvatures." *Angle Orthodontist*, 78(2):281-287, 2008.

### FEA for Clear Aligners
[35] J. Chen et al. "Three-dimensional FEA of the optimal mechanical design for anterior teeth movement with clear aligners." *Scientific Reports*, 14:63907, 2024. DOI: 10.1038/s41598-024-63907-x.

[36] "Materials for clear aligners: A comprehensive exploration." *Applied Sciences*, 14(15):6533, 2024. DOI: 10.3390/app14156533.

### Dental Processing Tools
[37] C. Lian et al. "Deep multi-scale mesh feature learning for automated labeling of raw dental surfaces (MeshSegNet)." *IEEE TMI*, 39(7):2440-2450, 2020. DOI: 10.1109/TMI.2020.2971730.

[38] "An implicit parametric morphable dental model (DMM)." *ACM TOG (SIGGRAPH Asia)*, 41(6):189, 2022. DOI: 10.1145/3550454.3555469.
