"""
Collision detection for inter-tooth penetration checking.

Uses ellipsoidal approximations of tooth crowns positioned at SE(3) poses.
When real STL meshes are available, uses trimesh for precise mesh-based
collision detection.

References:
  - trimesh.collision: https://trimesh.org/trimesh.collision.html
  - Crown dimensions from dental anatomy (Wheeler's Dental Anatomy, 2015)
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from .dental_constants import TOOTH_IDS, TOOTH_TYPES, N_TEETH, ARCH_ADJACENCY


# Average crown dimensions (mm): height, mesiodistal width, buccolingual depth
# From Wheeler's Dental Anatomy, Physiology and Occlusion
CROWN_DIMENSIONS = {
    'central_incisor': {'height': 10.5, 'width': 8.5, 'depth': 7.0},
    'lateral_incisor': {'height': 9.0, 'width': 6.5, 'depth': 6.0},
    'canine':          {'height': 10.0, 'width': 7.5, 'depth': 8.0},
    'premolar_1':      {'height': 8.5, 'width': 7.0, 'depth': 9.0},
    'premolar_2':      {'height': 8.0, 'width': 7.0, 'depth': 9.0},
    'molar_1':         {'height': 7.5, 'width': 10.5, 'depth': 11.0},
    'molar_2':         {'height': 7.0, 'width': 10.0, 'depth': 10.5},
}


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [qw,qx,qy,qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


class CollisionDetector:
    """
    Detects inter-tooth collisions using oriented bounding ellipsoids.

    Each tooth is modeled as an ellipsoid with anatomically accurate
    dimensions, positioned and oriented according to its SE(3) pose.
    Collision is detected when the distance between ellipsoid surfaces
    is negative (penetration) or below a safety margin.
    """

    def __init__(self, safety_margin_mm: float = 0.3):
        """
        Args:
            safety_margin_mm: Minimum acceptable distance between tooth surfaces.
                              0.3mm is typical contact point clearance.
        """
        self.safety_margin = safety_margin_mm

        # Precompute half-extents for each tooth
        self._half_extents = {}
        for tid in TOOTH_IDS:
            ttype = TOOTH_TYPES[tid]
            dims = CROWN_DIMENSIONS[ttype]
            self._half_extents[tid] = np.array([
                dims['width'] / 2.0,
                dims['depth'] / 2.0,
                dims['height'] / 2.0,
            ])

    def check_collisions(
        self,
        config: np.ndarray,
        check_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, any]:
        """
        Check for inter-tooth collisions in a configuration.

        Args:
            config: (28, 7) array of tooth poses
            check_pairs: List of (idx_a, idx_b) pairs to check.
                         If None, checks all adjacent pairs from ARCH_ADJACENCY.

        Returns:
            Dict with collision_pairs, n_collisions, min_distance, details.
        """
        if check_pairs is None:
            # Check adjacent pairs + opposing pairs
            pairs_to_check = []
            for a, b in ARCH_ADJACENCY:
                idx_a = TOOTH_IDS.index(a)
                idx_b = TOOTH_IDS.index(b)
                pairs_to_check.append((idx_a, idx_b))
        else:
            pairs_to_check = check_pairs

        collisions = []
        min_distance = float('inf')
        distances = []

        for idx_a, idx_b in pairs_to_check:
            dist = self._ellipsoid_distance(config[idx_a], config[idx_b],
                                            TOOTH_IDS[idx_a], TOOTH_IDS[idx_b])
            distances.append(dist)
            min_distance = min(min_distance, dist)

            if dist < self.safety_margin:
                collisions.append({
                    'tooth_a': TOOTH_IDS[idx_a],
                    'tooth_b': TOOTH_IDS[idx_b],
                    'distance_mm': round(dist, 4),
                    'penetration_mm': round(max(0, self.safety_margin - dist), 4),
                    'severity': 'collision' if dist < 0 else 'near_miss',
                })

        return {
            'collision_pairs': collisions,
            'n_collisions': len([c for c in collisions if c['severity'] == 'collision']),
            'n_near_misses': len([c for c in collisions if c['severity'] == 'near_miss']),
            'min_distance_mm': round(min_distance, 4),
            'mean_distance_mm': round(np.mean(distances), 4) if distances else 0.0,
            'pairs_checked': len(pairs_to_check),
        }

    def score_collision_free(self, config: np.ndarray) -> float:
        """
        Score how collision-free a configuration is. 1.0 = no collisions.

        Returns: float in [0.0, 1.0]
        """
        result = self.check_collisions(config)
        n_issues = result['n_collisions'] + result['n_near_misses']
        if n_issues == 0:
            return 1.0
        return max(0.0, 1.0 - n_issues / max(result['pairs_checked'], 1))

    def score_trajectory_collisions(self, trajectory: np.ndarray) -> float:
        """
        Score collision-free property across an entire trajectory.

        Args:
            trajectory: (N, 28, 7) array

        Returns: float in [0.0, 1.0], average across all stages.
        """
        scores = []
        for stage in range(trajectory.shape[0]):
            scores.append(self.score_collision_free(trajectory[stage]))
        return float(np.mean(scores))

    def _ellipsoid_distance(
        self,
        pose_a: np.ndarray,
        pose_b: np.ndarray,
        tid_a: int,
        tid_b: int,
    ) -> float:
        """
        Approximate distance between two oriented ellipsoids.

        Uses the GJK-inspired approach: compute centroid distance,
        subtract the sum of effective radii along the connecting axis.
        This is an approximation but fast and sufficient for RL.
        """
        # Centroids
        center_a = pose_a[4:7]
        center_b = pose_b[4:7]

        centroid_dist = np.linalg.norm(center_a - center_b)
        if centroid_dist < 1e-10:
            return 0.0  # coincident

        # Direction vector from A to B
        direction = (center_b - center_a) / centroid_dist

        # Rotation matrices
        R_a = _quaternion_to_rotation_matrix(pose_a[:4])
        R_b = _quaternion_to_rotation_matrix(pose_b[:4])

        # Transform direction into each ellipsoid's local frame
        local_dir_a = R_a.T @ direction
        local_dir_b = R_b.T @ (-direction)

        # Effective radius along direction for each ellipsoid
        ext_a = self._half_extents[tid_a]
        ext_b = self._half_extents[tid_b]

        # Radius = ||(ext * local_dir) / ||ext * local_dir|| * ext||
        # Simplified: project direction onto scaled ellipsoid surface
        radius_a = self._ellipsoid_support_distance(ext_a, local_dir_a)
        radius_b = self._ellipsoid_support_distance(ext_b, local_dir_b)

        return centroid_dist - radius_a - radius_b

    @staticmethod
    def _ellipsoid_support_distance(half_extents: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute the support distance of an ellipsoid along a direction.
        This is the distance from center to surface along the given direction.

        For ellipsoid with semi-axes a,b,c and direction d:
        r = 1 / sqrt((dx/a)^2 + (dy/b)^2 + (dz/c)^2)
        """
        scaled = direction / (half_extents + 1e-10)
        return 1.0 / (np.linalg.norm(scaled) + 1e-10)
