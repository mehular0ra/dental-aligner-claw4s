"""
Simplified Periodontal Ligament (PDL) spring model for biomechanically
grounded tooth movement simulation.

Models the PDL as a Kelvin-Voigt viscoelastic element (spring + dashpot)
with per-tooth-type stiffness based on root surface area.

References:
  - An Efficient Spring Model for Orthodontic Tooth Movement.
    Applied Sciences, 13(8), 5013 (2023). DOI: 10.3390/app13085013
  - Open-Full-Jaw: E_PDL = 68.9 MPa, v = 0.45
  - Thoeni et al. arXiv:2209.07576
"""
import math
import numpy as np
from typing import Dict

from .dental_constants import TOOTH_IDS, TOOTH_TYPES, N_TEETH


# Per-tooth-type PDL stiffness (N/mm) — derived from root surface area
# Molars have 2-3 roots with ~3-5x the surface area of single-rooted incisors
# Values calibrated from FEA literature (68.9 MPa modulus, 0.2mm PDL thickness)
PDL_STIFFNESS_TRANSLATION = {
    'central_incisor': 0.20,   # single root, small
    'lateral_incisor': 0.25,   # single root, small
    'canine': 0.35,            # single root, longest
    'premolar_1': 0.40,        # 1-2 roots
    'premolar_2': 0.45,        # 1-2 roots
    'molar_1': 0.70,           # 3 roots, large surface area
    'molar_2': 0.80,           # 3 roots, large surface area
}

# Rotational stiffness (N*mm/deg) — proportional to root length^2
PDL_STIFFNESS_ROTATION = {
    'central_incisor': 0.05,
    'lateral_incisor': 0.06,
    'canine': 0.10,
    'premolar_1': 0.08,
    'premolar_2': 0.08,
    'molar_1': 0.15,
    'molar_2': 0.18,
}

# Maximum safe force before PDL damage (N)
# Exceeding these causes root resorption in clinical practice
MAX_FORCE_TRANSLATION = {
    'central_incisor': 0.5,
    'lateral_incisor': 0.5,
    'canine': 0.75,
    'premolar_1': 0.75,
    'premolar_2': 0.75,
    'molar_1': 1.5,
    'molar_2': 1.5,
}

# Aligner material properties
ALIGNER_YOUNGS_MODULUS_MPA = 1361.0  # PETG (most common aligner material)
ALIGNER_POISSONS_RATIO = 0.30
ALIGNER_THICKNESS_MM = 0.75  # standard aligner shell thickness


class PDLModel:
    """
    Simplified spring-dashpot PDL model for tooth movement.

    Instead of teleporting teeth via SLERP, this model computes the
    biomechanically constrained displacement that an aligner can produce
    in one stage (~2 weeks of clinical wear).

    Usage:
        pdl = PDLModel()
        # Given desired delta, compute actual achievable delta
        actual_delta = pdl.constrain_movement(tooth_type, desired_delta_mm, desired_delta_deg)
        # Or compute the force required
        force = pdl.compute_required_force(tooth_type, delta_mm)
    """

    def constrain_movement(
        self,
        tooth_type: str,
        desired_trans_mm: float,
        desired_rot_deg: float,
    ) -> Dict[str, float]:
        """
        Given desired per-stage movement, compute biomechanically
        achievable movement accounting for PDL stiffness.

        Returns:
            Dict with actual_trans_mm, actual_rot_deg, force_n,
            torque_nmm, is_safe (within PDL damage threshold).
        """
        k_trans = PDL_STIFFNESS_TRANSLATION.get(tooth_type, 0.5)
        k_rot = PDL_STIFFNESS_ROTATION.get(tooth_type, 0.1)
        max_force = MAX_FORCE_TRANSLATION.get(tooth_type, 1.0)

        # Force required for desired translation
        force = k_trans * desired_trans_mm
        torque = k_rot * desired_rot_deg

        # Clamp to safe force limits
        if force > max_force:
            actual_trans = max_force / k_trans
            is_safe = False
        else:
            actual_trans = desired_trans_mm
            is_safe = True

        # Rotational limit (proportional to force limit)
        max_torque = max_force * 2.0  # simplified
        if torque > max_torque:
            actual_rot = max_torque / k_rot
            is_safe = False
        else:
            actual_rot = desired_rot_deg

        return {
            'actual_trans_mm': actual_trans,
            'actual_rot_deg': actual_rot,
            'force_n': min(force, max_force),
            'torque_nmm': min(torque, max_torque),
            'is_safe': is_safe,
            'efficiency': actual_trans / max(desired_trans_mm, 1e-6),
        }

    def compute_movement_resistance(self, config: np.ndarray) -> np.ndarray:
        """
        Compute per-tooth movement resistance factor (0-1).
        Higher = harder to move. Based on PDL stiffness.

        Returns: (28,) array of resistance values normalized to [0, 1].
        """
        resistances = np.zeros(N_TEETH, dtype=np.float64)
        max_k = max(PDL_STIFFNESS_TRANSLATION.values())
        for i, tid in enumerate(TOOTH_IDS):
            tooth_type = TOOTH_TYPES[tid]
            k = PDL_STIFFNESS_TRANSLATION.get(tooth_type, 0.5)
            resistances[i] = k / max_k
        return resistances

    def compute_force_budget(self, config: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute the total force budget needed to move all teeth to target.
        Useful for the agent to understand case complexity.
        """
        total_force = 0.0
        total_torque = 0.0
        unsafe_teeth = 0

        for i, tid in enumerate(TOOTH_IDS):
            tooth_type = TOOTH_TYPES[tid]
            k_trans = PDL_STIFFNESS_TRANSLATION.get(tooth_type, 0.5)

            trans_dist = float(np.linalg.norm(config[i, 4:7] - target[i, 4:7]))
            per_stage_trans = trans_dist / 24.0  # spread over 24 stages

            force = k_trans * per_stage_trans
            total_force += force

            max_force = MAX_FORCE_TRANSLATION.get(tooth_type, 1.0)
            if force > max_force:
                unsafe_teeth += 1

        return {
            'total_force_n': round(total_force, 4),
            'mean_force_per_tooth_n': round(total_force / N_TEETH, 4),
            'unsafe_teeth': unsafe_teeth,
            'force_feasibility': 1.0 - unsafe_teeth / N_TEETH,
        }

    def score_biomechanical_feasibility(
        self,
        trajectory: np.ndarray,
    ) -> float:
        """
        Score the biomechanical feasibility of a full trajectory.
        Penalizes movements that exceed safe PDL force limits.

        Args:
            trajectory: (26, 28, 7) array

        Returns:
            Score in [0.0, 1.0] where 1.0 = all movements within safe limits.
        """
        total_violations = 0
        total_checks = 0

        for stage in range(1, trajectory.shape[0]):
            prev = trajectory[stage - 1]
            curr = trajectory[stage]
            for i, tid in enumerate(TOOTH_IDS):
                tooth_type = TOOTH_TYPES[tid]
                delta_trans = float(np.linalg.norm(curr[i, 4:7] - prev[i, 4:7]))

                k = PDL_STIFFNESS_TRANSLATION.get(tooth_type, 0.5)
                force = k * delta_trans
                max_force = MAX_FORCE_TRANSLATION.get(tooth_type, 1.0)

                total_checks += 1
                if force > max_force:
                    total_violations += 1

        if total_checks == 0:
            return 1.0
        return 1.0 - total_violations / total_checks
