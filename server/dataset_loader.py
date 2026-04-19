"""
Dataset loaders for real clinical dental data.
Converts various dataset formats to the (28, 7) SE(3) pose representation
used by the dental aligner RL environment.

Each tooth pose is [qw, qx, qy, qz, tx, ty, tz]:
  - qw, qx, qy, qz: unit quaternion (scalar-first)
  - tx, ty, tz: translation in mm
"""
import json
import math
import os
import numpy as np
from glob import glob
from typing import Optional

from .dental_constants import TOOTH_IDS, N_TEETH
from .quaternion_utils import quaternion_normalize


# Map FDI tooth ID → array index (0-27)
TOOTH_ID_TO_INDEX = {tid: i for i, tid in enumerate(TOOTH_IDS)}

# UNN (Universal Numbering) to FDI mapping for Open-Full-Jaw
# UNN 1-16 = upper right to upper left, 17-32 = lower left to lower right
UNN_TO_FDI = {
    1: 18, 2: 17, 3: 16, 4: 15, 5: 14, 6: 13, 7: 12, 8: 11,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    17: 38, 18: 37, 19: 36, 20: 35, 21: 34, 22: 33, 23: 32, 24: 31,
    25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48,
}


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector, returning zero vector if near-zero."""
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return v / n


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz].
    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quaternion_normalize(q)


def _pca_rotation_matrix(vertices: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix from PCA of vertex positions.
    Returns a proper rotation matrix (det = +1).
    """
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    order = eigenvalues.argsort()[::-1]
    R = eigenvectors[:, order]
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


# ---------------------------------------------------------------------------
# Open-Full-Jaw loader
# ---------------------------------------------------------------------------

def load_open_full_jaw(patient_dir: str) -> np.ndarray:
    """
    Load SE(3) poses from Open-Full-Jaw JSON principal axes.

    Args:
        patient_dir: Path to extracted patient directory containing
                     teeth_principal_axes.json (or similar).

    Returns:
        (28, 7) numpy array of tooth poses.
    """
    config = np.zeros((N_TEETH, 7), dtype=np.float64)
    config[:, 0] = 1.0  # default identity quaternion

    # Find the teeth axes JSON file (e.g. teeth_axes_mandible.json, teeth_principal_axes.json)
    json_candidates = glob(os.path.join(patient_dir, "**/*teeth*axes*.json"), recursive=True)
    if not json_candidates:
        json_candidates = glob(os.path.join(patient_dir, "**/*principal*axes*.json"), recursive=True)
    if not json_candidates:
        raise FileNotFoundError(f"No teeth axes JSON found in {patient_dir}")

    with open(json_candidates[0]) as f:
        axes_data = json.load(f)

    # Format: dict keyed by UNN string, e.g. {"18": {"c": [...], "x": [...], "y": [...], "z": [...]}}
    # "c" = center of mass, "x"/"y"/"z" = axis endpoint (direction = endpoint - center)
    # Or could be a list of tooth entries with "UNN" field
    if isinstance(axes_data, dict) and all(k.isdigit() for k in axes_data):
        # Dict format: {"18": {"c": [...], "x": [...], ...}, "19": {...}, ...}
        tooth_entries = [(int(k), v) for k, v in axes_data.items()]
    elif isinstance(axes_data, list):
        tooth_entries = [(t.get("UNN", t.get("id", 0)), t) for t in axes_data]
    else:
        tooth_entries = [(int(k), v) for k, v in axes_data.get("teeth", {}).items()]

    for unn_id, tooth_data in tooth_entries:
        # Convert UNN to FDI (UNN 1-32 maps to FDI system)
        fdi_id = UNN_TO_FDI.get(unn_id)
        if fdi_id is None or fdi_id not in TOOTH_ID_TO_INDEX:
            continue  # wisdom tooth or invalid ID

        idx = TOOTH_ID_TO_INDEX[fdi_id]

        # Translation = center of mass ("c" key)
        com = tooth_data.get("c") or tooth_data.get("center_of_mass") or tooth_data.get("centroid")
        if com is None:
            continue
        center = np.array(com, dtype=np.float64)
        config[idx, 4:7] = center

        # Rotation from 3 axis endpoints relative to center
        # "x", "y", "z" are endpoint positions; direction = endpoint - center
        x_pt = tooth_data.get("x")
        y_pt = tooth_data.get("y")
        z_pt = tooth_data.get("z")

        if x_pt is not None and y_pt is not None and z_pt is not None:
            x_dir = _normalize_vec(np.array(x_pt, dtype=np.float64) - center)
            y_dir = _normalize_vec(np.array(y_pt, dtype=np.float64) - center)
            z_dir = _normalize_vec(np.array(z_pt, dtype=np.float64) - center)

            # Build rotation matrix and orthogonalize via SVD
            R = np.column_stack([x_dir, y_dir, z_dir])
            U, _, Vt = np.linalg.svd(R)
            R_ortho = U @ Vt
            if np.linalg.det(R_ortho) < 0:
                U[:, 2] *= -1
                R_ortho = U @ Vt

            config[idx, 0:4] = rotation_matrix_to_quaternion(R_ortho)

    return config


# ---------------------------------------------------------------------------
# Teeth3DS+ loader
# ---------------------------------------------------------------------------

def load_teeth3ds(obj_path: str, json_path: str) -> np.ndarray:
    """
    Load SE(3) poses from Teeth3DS+ OBJ mesh + JSON labels.

    Args:
        obj_path: Path to .obj mesh file.
        json_path: Path to .json annotation file with per-vertex FDI labels.

    Returns:
        (28, 7) numpy array of tooth poses.
    """
    config = np.zeros((N_TEETH, 7), dtype=np.float64)
    config[:, 0] = 1.0

    # Load mesh vertices (minimal OBJ parser — no trimesh dependency)
    vertices = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices, dtype=np.float64)

    # Load labels
    with open(json_path) as f:
        label_data = json.load(f)

    labels = np.array(label_data.get("labels", label_data.get("tooth_labels", [])))

    if len(labels) != len(vertices):
        return config  # shape mismatch, return identity

    for fdi_id in TOOTH_IDS:
        if fdi_id not in TOOTH_ID_TO_INDEX:
            continue
        idx = TOOTH_ID_TO_INDEX[fdi_id]
        mask = labels == fdi_id
        if not mask.any():
            continue

        tooth_verts = vertices[mask]
        if len(tooth_verts) < 10:
            continue  # too few points for reliable PCA

        # Translation = centroid
        config[idx, 4:7] = tooth_verts.mean(axis=0)

        # Rotation from PCA
        R = _pca_rotation_matrix(tooth_verts)
        config[idx, 0:4] = rotation_matrix_to_quaternion(R)

    return config


# ---------------------------------------------------------------------------
# Mendeley Jaw Models loader
# ---------------------------------------------------------------------------

def load_mendeley_jaw(teeth_dir: str) -> np.ndarray:
    """
    Load SE(3) poses from Mendeley Jaw Models (pre-segmented STL files).
    Only lower arch (FDI 31-37, 41-47 = 14 teeth). Upper arch set to identity.

    Args:
        teeth_dir: Path to Teeth/ directory containing D37.stl, D38.stl, etc.

    Returns:
        (28, 7) numpy array of tooth poses.
    """
    config = np.zeros((N_TEETH, 7), dtype=np.float64)
    config[:, 0] = 1.0

    stl_files = glob(os.path.join(teeth_dir, "D*.stl")) + glob(os.path.join(teeth_dir, "d*.stl"))

    for stl_path in stl_files:
        # Extract FDI ID from filename: D37.stl → 37
        basename = os.path.basename(stl_path)
        try:
            fdi_id = int("".join(c for c in basename if c.isdigit()))
        except ValueError:
            continue

        if fdi_id not in TOOTH_ID_TO_INDEX:
            continue

        idx = TOOTH_ID_TO_INDEX[fdi_id]

        # Minimal binary STL reader (no trimesh dependency)
        vertices = _read_stl_vertices(stl_path)
        if vertices is None or len(vertices) < 10:
            continue

        config[idx, 4:7] = vertices.mean(axis=0)
        R = _pca_rotation_matrix(vertices)
        config[idx, 0:4] = rotation_matrix_to_quaternion(R)

    return config


def _read_stl_vertices(stl_path: str) -> Optional[np.ndarray]:
    """Read vertices from a binary STL file (no external dependencies)."""
    try:
        with open(stl_path, "rb") as f:
            f.read(80)  # header
            n_triangles = int.from_bytes(f.read(4), byteorder="little")
            vertices = []
            for _ in range(n_triangles):
                f.read(12)  # normal vector
                for _ in range(3):  # 3 vertices per triangle
                    vx = np.frombuffer(f.read(4), dtype=np.float32)[0]
                    vy = np.frombuffer(f.read(4), dtype=np.float32)[0]
                    vz = np.frombuffer(f.read(4), dtype=np.float32)[0]
                    vertices.append([vx, vy, vz])
                f.read(2)  # attribute byte count
            return np.array(vertices, dtype=np.float64)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_SOURCES = {
    "open_full_jaw": {
        "name": "Open-Full-Jaw",
        "description": "17 patients with JSON principal axes + FEM meshes",
        "license": "CC BY-NC-SA 4.0",
        "n_patients": 17,
        "loader": "load_open_full_jaw",
    },
    "teeth3ds": {
        "name": "Teeth3DS+",
        "description": "1,800 intraoral scans with per-vertex FDI segmentation",
        "license": "CC BY-NC-ND 4.0",
        "n_patients": 900,
        "loader": "load_teeth3ds",
    },
    "mendeley_jaw": {
        "name": "Mendeley Jaw Models",
        "description": "1 patient, 14 pre-segmented lower teeth + PDL",
        "license": "CC BY 4.0",
        "n_patients": 1,
        "loader": "load_mendeley_jaw",
    },
}


def list_datasets() -> dict:
    """Return metadata about available dataset sources."""
    return DATASET_SOURCES
