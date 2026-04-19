# Data Acquisition Plan — Dental Aligner RL Environment

## Context

Our RL environment for orthodontic aligner trajectory planning needs 3D dental data (pre/post-treatment pairs) to define realistic start/goal states and calibrate reward signals. This document outlines all viable data sources, synthetic generation approaches, and a prioritized action plan.

---

## 1. Public Datasets

### 1.1 Datasets with Pre/Post-Treatment Pairs (Most Valuable)

#### Tsinghua 3D Orthodontic Dental Dataset
- **Samples:** 1,060 pre/post-treatment pairs from 435 patients (ages 8-35)
- **Contents:** STL meshes, tooth segmentation labels, tooth position transforms, crown landmarks, malocclusion categories (crowding, deep overbite, deep overjet)
- **Format:** STL + JSON annotations
- **Access:** Data Use Agreement required. Sign and email to Prof. Yong-Jin Liu (liuyongjin@tsinghua.edu.cn)
- **License:** Academic research only, no commercial use, no redistribution
- **Status:** DUA prepared, pending signature and submission
- **Links:**
  - Dataset: https://zenodo.org/records/11392406
  - Paper: https://www.nature.com/articles/s41597-024-04138-7
  - DUA template: https://github.com/lcshhh/TADPM/blob/main/Data-Use-Agreement.pdf
  - Source code: https://github.com/lcshhh/TADPM

#### CLIK-Diffusion Dataset
- **Samples:** Real-world clinical malocclusion cases with landmark transformations
- **Contents:** Pre/post treatment pairs with hierarchical clinical constraints (dental-arch level, inter-tooth level, individual-tooth level) based on Andrews' Six Keys
- **Format:** Point clouds + landmark coordinates
- **Access:** Data Access Agreement, research purpose only
- **Links:**
  - Code + data access: https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion
  - Paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841525002932

#### TeethAlign3D (ICCV 2025)
- **Samples:** 855 clinical cases with transformation matrices
- **Contents:** 3D dental models with tooth alignment target transformation matrices, occlusion and collision constraint information
- **Access:** Dataset and code release announced, check project page for updates
- **Links:**
  - Project page: https://californiachen.github.io/publications/2025ICCV/
  - Dataset page: https://californiachen.github.io/datasets/
  - Paper: https://arxiv.org/abs/2410.20806

#### TADPM / TeethGenerator Dataset
- **Samples:** 212 pairs (original TADPM); 720 train / 80 val / 120 test (expanded)
- **Contents:** Pre/post-treatment dental models with tooth transformation matrices, point cloud and mesh features
- **Access:** Request via GitHub
- **Links:**
  - TADPM code: https://github.com/lcshhh/TADPM
  - TeethGenerator code: https://github.com/lcshhh/teeth_generator
  - TADPM paper: https://www.sciencedirect.com/science/article/abs/pii/S016783962400027X
  - TeethGenerator paper: https://arxiv.org/abs/2507.04685

### 1.2 Intraoral Scan Datasets (No Treatment Pairs, but Useful Geometry)

#### Teeth3DS+
- **Samples:** 1,800 intraoral scans from 900 patients, 23,999 annotated teeth
- **Contents:** OBJ meshes + JSON segmentation labels (FDI numbering). Subset of 340 scans has dental landmarks
- **Scanners used:** Primescan, TRIOS 3, iTero Element 2 Plus
- **Access:** Free registration required. Also available via PyTorch Geometric: `torch_geometric.datasets.Teeth3DS`
- **Why useful:** Extract realistic tooth shapes and per-tooth poses; re-pose teeth to create synthetic malocclusion scenarios
- **Links:**
  - Website: https://crns-smartvision.github.io/teeth3ds/
  - Data download: https://osf.io/xctdy/
  - Grand Challenge: https://3dteethseg.grand-challenge.org/
  - Paper: https://arxiv.org/abs/2210.06094

#### Bits2Bites
- **Samples:** 200 paired upper/lower arch STLs aligned in RAS coordinate system
- **Contents:** Multi-label occlusion classification (sagittal, vertical, transverse dimensions)
- **Access:** Account required
- **Why useful:** Paired arch scans with occlusion labels — directly relevant for occlusion constraint modeling
- **Links:**
  - Dataset: https://ditto.ing.unimore.it/bits2bites/

#### 3D-IOSSeg
- **Samples:** 180 clinical intraoral scans (120 train / 60 test) from 200+ patients
- **Contents:** Fine-grained mesh-unit-level tooth segmentation. Includes dental anomalies: missing teeth, overlapping, misalignment
- **Access:** Available via GitHub
- **Links:**
  - Code + data: https://github.com/MIVRC/Fast-TGCN
  - Paper: https://www.sciencedirect.com/science/article/abs/pii/S0010482523012866

#### Poseidon3D
- **Samples:** 200 meshes (88 maxillas, 122 mandibles), ~200k faces per mesh
- **Contents:** Challenging orthodontic cases: diastema, crowding, infraocclusion, missing/damaged teeth
- **Links:**
  - Paper: https://www.mdpi.com/2306-5354/11/10/1014

### 1.3 Anatomical / FEM Datasets

#### Open-Full-Jaw
- **Samples:** 17 patient-specific FEM models from CBCT
- **Contents:** Mandible, maxilla, teeth, periodontal ligament (PDL) meshes, teeth principal axes. Includes Python pipeline for generating FEM models
- **Access:** Fully open, immediate download
- **Why useful:** PDL meshes enable biomechanically realistic tooth movement simulation; teeth principal axes provide SE(3) pose information
- **Links:**
  - Code + data: https://github.com/diku-dk/Open-Full-Jaw
  - Paper: https://arxiv.org/abs/2209.07576

#### Mendeley Synthetic Jaw Models
- **Contents:** Complete jaw with 14 teeth + 14 periodontal ligaments. STL, IGES, SEDOC formats
- **Access:** Open
- **Links:**
  - Dataset 1: https://data.mendeley.com/datasets/xjsx7nfhj8/1
  - Dataset 2: https://data.mendeley.com/datasets/97c3pf8h2h/1

#### Serial 3D Dental Models (Longitudinal)
- **Samples:** 24 patients scanned at different time intervals during growth
- **Contents:** Upper and lower dentitions, normal occlusion, no orthodontic treatment
- **Access:** Open
- **Why useful:** Shows natural tooth movement trajectories over time
- **Links:**
  - Dataset: https://data.mendeley.com/datasets/bpnmf2vhsk/1

### 1.4 CBCT / Volumetric Datasets

| Dataset | Samples | Access | Link |
|---------|---------|--------|------|
| ToothFairy 1 | 443 CBCTs, IAC annotations | CC BY-SA | https://ditto.ing.unimore.it/toothfairy/ |
| ToothFairy 2 | 42-class multi-structure seg | CC BY-SA | https://ditto.ing.unimore.it/toothfairy2/ |
| ToothFairy 3 | 532 CBCTs, 77-class annotation | CC BY-NC-SA | https://ditto.ing.unimore.it/toothfairy3/ |
| CTooth+ | 22 annotated + 146 unlabeled CBCTs | Kaggle | https://www.kaggle.com/datasets/weiweicui/ctooth-dataset |
| STS-Tooth | 4,000 X-rays + 148,400 CBCT slices | Zenodo (open) | https://zenodo.org/records/10597292 |
| MMDental | 660 patients, CBCTs + medical records | Figshare (open) | https://springernature.figshare.com/articles/dataset/MMDental_-_A_multimodal_dataset_of_tooth_CBCT_images_with_expert_medical_records/28505276 |
| 3D Multimodal (CBCT+Oral) | 289 paired CBCT + oral scan | Figshare | https://figshare.com/articles/dataset/_b_3D_multimodal_dental_dataset_based_on_CBCT_and_oral_scan_b_/26965903 |
| PhysioNet Dental | 329 CBCTs from 169 patients | Credentialed access | https://physionet.org/content/multimodal-dental-dataset/1.1.0/ |
| MedShapeNet | 100k+ medical shapes (includes dental) | Open | https://medshapenet.ikim.nrw/ |

### 1.5 Segmentation Tools (for processing raw scans)

| Tool | What It Does | Link |
|------|-------------|------|
| DentalSegmentator | Auto-segment maxilla, mandible, teeth from CBCT | https://github.com/gaudot/SlicerDentalSegmentator |
| MeshSegNet | Tooth segmentation on intraoral scan meshes (PyTorch) | https://github.com/Tai-Hsien/MeshSegNet |
| ToothSeg (DKFZ) | Deep learning tooth segmentation from CBCT | https://github.com/MIC-DKFZ/ToothSeg |

---

## 2. Synthetic Data Generation

### 2.1 Mathematical Arch Curves (Lowest Effort, No Data Needed)

Generate realistic dental arch shapes programmatically and place tooth poses along them.

**Approach:**
- Use beta functions, catenary curves, or polynomial functions (Y = AX^6 + BX^2) to define arch shape
- Place 28 tooth poses along the arch with anatomically-informed spacing
- Generate malocclusions by applying controlled SE(3) perturbations grouped by difficulty
- Maps directly to our existing `[qw, qx, qy, qz, tx, ty, tz]` per-tooth representation

**References:**
- Beta function arch model: https://link.springer.com/article/10.1007/s10266-016-0244-7
- Personalized arch forms: https://link.springer.com/article/10.1186/s12903-025-06993-1

**Pros:** Immediate, unlimited data, pure Python, no external dependencies
**Cons:** May not capture real-world anatomical variation; needs clinical validation

### 2.2 TeethGenerator (Paired Synthetic Pre/Post Data)

Two-stage framework generating paired pre- and post-orthodontic 3D teeth.

**Stage 1:** VQ-VAE + diffusion model generates diverse post-orthodontic teeth point clouds
**Stage 2:** Transformer takes style information to generate corresponding pre-orthodontic teeth (simulating malocclusion from well-aligned teeth)

- Code: https://github.com/lcshhh/teeth_generator
- Paper: https://arxiv.org/abs/2507.04685

**Pros:** Generates paired data validated against real clinical distributions
**Cons:** Requires Tsinghua dataset to train the generator; outputs point clouds (PLY), not SE(3) poses directly

### 2.3 DMM — Implicit Parametric Morphable Dental Model

First parametric 3D morphable model for teeth + gums using implicit neural representations (DeepSDF-based).

- Sample latent space to generate diverse tooth configurations
- Pretrained models available on Google Drive
- Code: https://github.com/cong-yi/DMM
- Project page: https://vcai.mpi-inf.mpg.de/projects/DMM/
- Paper: https://arxiv.org/abs/2211.11402

**Pros:** High-fidelity generation; latent space interpolation maps to generating malocclusion-to-target pairs
**Cons:** Requires SDF preprocessing; trained on private scan dataset

### 2.4 ToothForge — Spectral Shape Generation

Generates high-resolution 3D tooth meshes in milliseconds using synchronized spectral embeddings.

- Code: https://github.com/tibirkubik/toothForge
- Paper: https://arxiv.org/abs/2506.02702

**Pros:** Very fast generation; high quality
**Cons:** Individual teeth only, not full arches; needs GPU (8GB+ VRAM)

### 2.5 Spring-Mass PDL Model (Physics Simulation)

Model periodontal ligament as springs connecting teeth to alveolar bone. System energy relaxes by changing tooth position/orientation toward equilibrium.

- Reference: https://www.mdpi.com/2076-3417/13/8/5013
- FEA review: https://arxiv.org/abs/2503.24195

**Pros:** Lightweight physics that captures real biomechanics; could replace pure geometric SLERP interpolation as environment dynamics
**Cons:** Needs parameter tuning; simplified compared to full FEA

### 2.6 Other Generative Models

| Model | What | Status | Link |
|-------|------|--------|------|
| DuoDent | Dual-stream diffusion for tooth point clouds (MICCAI 2025) | Code "under preparation" | https://github.com/kdy-ku/DuoDent |
| DM-CFO | Diffusion with collision-free optimization | Very recent (Mar 2026) | https://arxiv.org/abs/2603.03602 |
| Tooth-Diffusion | 3D conditional diffusion for dental CBCT | Open source | https://github.com/djafar1/tooth-diffusion |

---

## 3. Getting Data from a Dentist

### 3.1 What We Need Per Patient

| File | Format | Typical Size |
|------|--------|-------------|
| Pre-treatment upper arch | STL | 5-50 MB |
| Pre-treatment lower arch | STL | 5-50 MB |
| Post-treatment upper arch | STL | 5-50 MB |
| Post-treatment lower arch | STL | 5-50 MB |

**4 STL files per patient. No clinical notes, no X-rays, no demographics needed.**

### 3.2 Scanner Compatibility

| Scanner | Export Formats | Research-Friendliness | Notes |
|---------|--------------|:---:|-------|
| **Medit i700/i500** | STL, PLY, OBJ | Best | Fully open, no subscription locks, 2-min export per case |
| **3Shape TRIOS** | STL, PLY, OBJ, DCM | Good | Direct STL export from TRIOS software |
| **Primescan (Dentsply)** | STL, PLY | Good | Export from CEREC/inLab |
| **Carestream 3700** | STL, PLY | Good | Direct export |
| **iTero (Align Tech)** | STL (via MyAligntech) | Restrictive | Must have been scanned as iCast/iRecord, NOT Invisalign mode |

### 3.3 Privacy / Anonymization

STL files contain pure geometry (vertices + faces). No patient info is embedded in the mesh data. Anonymization requires only:
1. Export as STL (not DICOM)
2. Rename files to numeric IDs (remove patient names from filenames)
3. Remove any companion files with metadata

This is sufficient for research under a Data Use Agreement. Full-face photos or DICOM files require more careful de-identification, but we don't need those.

### 3.4 How to Approach

**Best targets:**
- Orthodontists with completed aligner cases in their scanner archive
- University dental school orthodontic departments
- Practices using Medit or TRIOS scanners (easiest export)

**What to offer:**
- Co-authorship on CLAW4S publication
- Small consulting fee ($500-1000 for their time)
- A simple 1-page Data Use Agreement

**What they do:** Export 10-20 completed cases as anonymized STL files from their archive. ~30 minutes of their time.

### 3.5 Cost Estimate (If Commissioning Scans)

- Practice cost per patient: ~$21-38
- For pre/post pairs, need patients who have actually completed treatment (can't just scan random people twice)
- Realistic budget for 20 cases from a private practice: $1,000-3,000
- Better approach: find a practice with existing archived cases

### 3.6 Dental Data Networks / Consortiums

| Network | Focus | 3D Data? | Link |
|---------|-------|:---:|------|
| BigMouth (UTHealth) | 7M+ patient EHRs from 17 dental schools | Planned | https://www.uth.edu/bigmouth/ |
| COHRI | 20+ dental school consortium | No (structured data) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3114442/ |
| NIDCR Data Hub | Aggregates dental research data sources | Links to FaceBase, BigMouth | https://www.ddshub.nih.gov/ |
| FaceBase | Craniofacial development data | Yes (3D facial norms) | https://www.facebase.org/ |

---

## 4. Key Papers and Related Work

### 4.1 Tooth Alignment / Arrangement

| Paper | Year | Data Used | Key Insight | Link |
|-------|:---:|-----------|-------------|------|
| TANet (ECCV) | 2020 | Private clinical | Pioneered learning-based tooth arrangement as 6-DOF pose prediction | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600477.pdf |
| PSTN (MICCAI) | 2020 | 304 private cases | PointNet-based spatial transformation network | https://link.springer.com/chapter/10.1007/978-3-030-59716-0_11 |
| iOrthoPredictor (SIGGRAPH Asia) | 2020 | 591 private cases | Hybrid 2D photo + 3D model approach | https://github.com/Lingchen-chen/iOrthoPredictor |
| TADPM (CAGD) | 2024 | 212 pairs | Diffusion model for tooth transforms; dental arch curve metric | https://github.com/lcshhh/TADPM |
| CLIK-Diffusion (MedIA) | 2025 | Private + released | Landmark-based formulation with hierarchical clinical constraints | https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion |
| Transformer Alignment (ICCV) | 2025 | 855 cases | 3D-to-2D texture projection; occlusion loss functions | https://arxiv.org/abs/2410.20806 |
| TAlignDiff | 2025 | Tsinghua 1,060 | Diffusion-based transformation refinement | https://arxiv.org/abs/2508.04565 |
| Deep Align Net | 2025 | Tsinghua 1,060 | End-to-end system with stage-wise treatment planning | https://link.springer.com/article/10.1007/s11633-025-1556-2 |
| auto_tooth_arrangement | — | Real dental data | Transformer-based relative pose on dental arch (pretrained weights available) | https://github.com/Otrho3D/auto_tooth_arrangement |

### 4.2 RL in Orthodontics

| Paper | Year | Data Used | Key Insight | Link |
|-------|:---:|-----------|-------------|------|
| Multi-task RL for Orthodontic-Orthognathic Treatment | 2025 | 347 patients | RL for sequential treatment decisions; 19.9% quality improvement, 73.9% time reduction | https://www.nature.com/articles/s41598-025-09236-z |
| AI-Driven Dynamic Orthodontic Treatment | 2025 | Review | Digital twin modeling for simulated orthodontic forces | https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2025.1612441/full |

### 4.3 Creative Alternatives to Full 3D Data

| Approach | Paper | Insight | Link |
|----------|-------|---------|------|
| 2D contour projection | BMVC 2023 (ShanghaiTech) | Project 3D tooth structures to 2D contours; learn alignment in 2D | https://github.com/ShanghaiTech-IMPACT/3D-Structure-guided-Network-for-Tooth-Alignment-in-2D-Photograph |
| 2D texture maps | ICCV 2025 | Convert 3D point clouds along arch to multi-channel textures | https://arxiv.org/abs/2410.20806 |
| Sparse landmarks | CLIK-Diffusion | ~4-6 landmarks per tooth x 28 teeth = ~504 numbers as state | https://github.com/ShanghaiTech-IMPACT/CLIK-Diffusion |
| 3D from 5 photos | TeethDreamer (MICCAI 2024) | Reconstruct full 3D teeth from just 5 intraoral photos | https://github.com/ShanghaiTech-IMPACT/TeethDreamer |
| Spring-mass PDL | MDPI 2023 | Lightweight biomechanical sim for tooth movement | https://www.mdpi.com/2076-3417/13/8/5013 |

### 4.4 FEA / Biomechanics

| Resource | What | Link |
|----------|------|------|
| Computational Orthodontic Force Simulation Review | Comprehensive FEA survey (CBCT -> STL -> FEM pipeline) | https://arxiv.org/abs/2503.24195 |
| 3D FEA for Clear Aligner Movement | Detailed material properties for teeth, PDL, bone, aligner | https://www.nature.com/articles/s41598-024-63907-x |

---

## 5. Open-Source Tools

| Tool | Purpose | Link |
|------|---------|------|
| MeshSegNet | Tooth segmentation on intraoral scans (PyTorch) | https://github.com/Tai-Hsien/MeshSegNet |
| DentalSegmentator | Auto-segment teeth/jaw from CBCT (3D Slicer plugin) | https://github.com/gaudot/SlicerDentalSegmentator |
| ToothSeg (DKFZ) | Deep learning tooth segmentation from CBCT | https://github.com/MIC-DKFZ/ToothSeg |
| BlenderForDental | Blender addons for dental workflows | https://www.blenderfordental.com/ |
| BDENTAL4D | Blender addon for dental DICOM/mesh processing | https://github.com/issamdakir/BDENTAL4D-WIN |
| Open Dental CAD | Blender-based open dental CAD | https://github.com/patmo141/odc_public |

---

## 6. Action Plan (Prioritized)

### Immediate (Today)

- [ ] Download **Teeth3DS+** from https://osf.io/xctdy/ (free, registration only)
- [ ] Download **Open-Full-Jaw** from https://github.com/diku-dk/Open-Full-Jaw (fully open)
- [ ] Implement **parametric arch curve generation** using beta/catenary functions — unlimited synthetic training data with no external dependencies

### This Week

- [ ] Sign and send the **Tsinghua DUA** to liuyongjin@tsinghua.edu.cn (both copies: Mehul + Vivek)
- [ ] Apply for **CLIK-Diffusion dataset** via their GitHub
- [ ] Check **TeethAlign3D** (ICCV 2025) dataset release status
- [ ] Send data request letter to any orthodontist contacts (we need just 10-20 anonymized STL case pairs)

### When Access Arrives

- [ ] Integrate Tsinghua pre/post pairs as ground truth for reward calibration
- [ ] Extract per-tooth SE(3) poses from real data to validate synthetic distributions
- [ ] Train TeethGenerator on real data for augmented synthetic generation
- [ ] Add required citations to README and all publications:
  - Wang et al. (2024) Scientific Data, DOI: 10.1038/s41597-024-04138-7
  - Lei et al. (2024) CAGD, DOI: 10.1016/j.cagd.2024.102293

### Stretch Goals

- [ ] Implement spring-mass PDL model as environment dynamics (replace pure SLERP interpolation)
- [ ] Try DMM parametric model for diverse synthetic tooth configurations
- [ ] Explore 2D contour projection as simplified state representation
- [ ] Contact university dental school orthodontic departments for larger-scale collaboration

---

## 7. Required Citations

Any publication using the Tsinghua dataset must cite:

> [1] Shaofeng Wang, Changsong Lei, Yaqian Liang, Jun Sun, Xianju Xie, Yajie Wang, Feifei Zuo, Yuxin Bai, Song Li & Yong-Jin Liu. A 3D dental model dataset with pre/post-orthodontic treatment for automatic tooth alignment. Scientific Data, Vol. 11, Article 1277 (2024). DOI: 10.1038/s41597-024-04138-7

> [2] Changsong Lei, Mengfei Xia, Shaofeng Wang, Yaqian Liang, Ran Yi, Yu-Hui Wen & Yong-Jin Liu. Automatic tooth arrangement with joint features of point and mesh representations via diffusion probabilistic models. Computer Aided Geometric Design, Vol. 111, 102293 (2024). DOI: 10.1016/j.cagd.2024.102293
