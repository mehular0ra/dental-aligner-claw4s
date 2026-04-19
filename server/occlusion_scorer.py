"""
Occlusion scoring based on Andrews' Six Keys to Normal Occlusion (1972)
and ABO Objective Grading System criteria.

All metrics are computable directly from the 28x7 SE(3) pose representation
without requiring mesh geometry.

References:
  Andrews LF. The six keys to normal occlusion. Am J Orthod. 1972;62(3):296-309.
  ABO Objective Grading System: americanboardortho.com
"""
import math
import numpy as np
from typing import Dict, List, Tuple

from .dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH, OPPOSING_PAIRS, ARCH_ADJACENCY,
)


# Andrews' ideal crown angulations (mesiodistal tip, degrees)
# Positive = mesial crown tip toward occlusal
IDEAL_ANGULATION = {
    'central_incisor': 5.0,
    'lateral_incisor': 9.0,
    'canine': 11.0,
    'premolar_1': 2.0,
    'premolar_2': 2.0,
    'molar_1': 5.0,
    'molar_2': 5.0,
}

# Andrews' ideal crown inclination (labiolingual torque, degrees)
# Positive = labial crown torque for uppers, lingual for lowers
IDEAL_INCLINATION_UPPER = {
    'central_incisor': 7.0,
    'lateral_incisor': 3.0,
    'canine': -7.0,
    'premolar_1': -7.0,
    'premolar_2': -7.0,
    'molar_1': -9.0,
    'molar_2': -9.0,
}

IDEAL_INCLINATION_LOWER = {
    'central_incisor': -1.0,
    'lateral_incisor': -1.0,
    'canine': -11.0,
    'premolar_1': -17.0,
    'premolar_2': -22.0,
    'molar_1': -30.0,
    'molar_2': -35.0,
}

# Average mesiodistal crown widths (mm) for Bolton analysis
CROWN_WIDTHS = {
    'central_incisor': 8.5,
    'lateral_incisor': 6.5,
    'canine': 7.5,
    'premolar_1': 7.0,
    'premolar_2': 6.5,
    'molar_1': 10.0,
    'molar_2': 9.5,
}


def _quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [qw,qx,qy,qz] to Euler angles (roll, pitch, yaw) in degrees.
    Roll = rotation around x-axis (labiolingual torque)
    Pitch = rotation around y-axis (mesiodistal tip)
    Yaw = rotation around z-axis (long axis rotation)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Roll (x-axis)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.degrees(math.atan2(sinr, cosr))

    # Pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    # Yaw (z-axis)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw


class OcclusionScorer:
    """
    Computes clinically grounded occlusion metrics from 28x7 SE(3) poses.
    All scores are in [0.0, 1.0] where 1.0 = ideal occlusion.
    """

    def score_all(self, config: np.ndarray) -> Dict[str, float]:
        """Compute all occlusion metrics. Returns dict of scores."""
        return {
            'molar_relationship': self.score_molar_relationship(config),
            'overjet': self.score_overjet(config),
            'overbite': self.score_overbite(config),
            'crown_angulation': self.score_crown_angulation(config),
            'crown_inclination': self.score_crown_inclination(config),
            'rotations': self.score_rotations(config),
            'contact_tightness': self.score_contact_tightness(config),
            'curve_of_spee': self.score_curve_of_spee(config),
            'arch_symmetry': self.score_arch_symmetry(config),
        }

    def score_composite(self, config: np.ndarray) -> float:
        """Weighted composite occlusion score."""
        scores = self.score_all(config)
        weights = {
            'molar_relationship': 0.20,
            'overjet': 0.15,
            'overbite': 0.10,
            'crown_angulation': 0.10,
            'crown_inclination': 0.10,
            'rotations': 0.10,
            'contact_tightness': 0.10,
            'curve_of_spee': 0.05,
            'arch_symmetry': 0.10,
        }
        return sum(scores[k] * weights[k] for k in weights)

    def score_molar_relationship(self, config: np.ndarray) -> float:
        """
        Andrews Key 1: Molar relationship (Angle Classification).
        Class I = upper first molar mesial cusp aligns with lower first molar buccal groove.
        Score based on anteroposterior discrepancy between opposing first molars.
        """
        scores = []
        # Right side: tooth 16 (idx 5) vs 46 (idx 26)
        # Left side: tooth 26 (idx 12) vs 36 (idx 19)
        molar_pairs = [(5, 26), (12, 19)]  # (upper idx, lower idx)

        for u_idx, l_idx in molar_pairs:
            upper_ty = config[u_idx, 5]  # anteroposterior position
            lower_ty = config[l_idx, 5]
            # Class I: upper molar slightly posterior to lower (~2mm offset)
            discrepancy = abs((upper_ty - lower_ty) - 2.0)
            score = max(0.0, 1.0 - discrepancy / 5.0)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def score_overjet(self, config: np.ndarray) -> float:
        """
        Overjet: anteroposterior distance between upper and lower incisors.
        Ideal: 2-3mm. Penalty for negative (underbite) or excessive (>5mm).
        """
        # Upper central incisors: teeth 11 (idx 0), 21 (idx 7)
        # Lower central incisors: teeth 41 (idx 21), 31 (idx 14)
        upper_ty = (config[0, 5] + config[7, 5]) / 2  # mean upper incisor Y
        lower_ty = (config[21, 5] + config[14, 5]) / 2  # mean lower incisor Y
        overjet = upper_ty - lower_ty  # positive = normal

        # Ideal range: 2-3mm
        if 2.0 <= overjet <= 3.0:
            return 1.0
        elif overjet < 0:
            return max(0.0, 1.0 + overjet / 5.0)  # negative overjet = underbite
        elif overjet > 3.0:
            return max(0.0, 1.0 - (overjet - 3.0) / 5.0)
        else:  # 0-2mm
            return 0.5 + overjet / 4.0

    def score_overbite(self, config: np.ndarray) -> float:
        """
        Overbite: vertical overlap of upper and lower incisors.
        Ideal: 2-3mm. Penalty for open bite (<0) or deep bite (>4mm).
        """
        upper_tz = (config[0, 6] + config[7, 6]) / 2
        lower_tz = (config[21, 6] + config[14, 6]) / 2
        overbite = upper_tz - lower_tz  # positive = normal overlap

        if 2.0 <= overbite <= 3.0:
            return 1.0
        elif overbite < 0:
            return max(0.0, 1.0 + overbite / 4.0)  # open bite
        elif overbite > 4.0:
            return max(0.0, 1.0 - (overbite - 4.0) / 4.0)  # deep bite
        else:
            return 0.7

    def score_crown_angulation(self, config: np.ndarray) -> float:
        """
        Andrews Key 2: Mesiodistal crown angulation (tip).
        Compare pitch angle of each tooth to Andrews' ideal values.
        """
        deviations = []
        for i, tid in enumerate(TOOTH_IDS):
            tooth_type = TOOTH_TYPES[tid]
            ideal = IDEAL_ANGULATION.get(tooth_type, 0.0)
            _, pitch, _ = _quaternion_to_euler(config[i, :4])
            deviation = abs(pitch - ideal)
            deviations.append(deviation)

        mean_deviation = np.mean(deviations)
        return max(0.0, 1.0 - mean_deviation / 15.0)

    def score_crown_inclination(self, config: np.ndarray) -> float:
        """
        Andrews Key 3: Labiolingual crown inclination (torque).
        Compare roll angle to Andrews' ideal values, different for upper/lower.
        """
        deviations = []
        for i, tid in enumerate(TOOTH_IDS):
            tooth_type = TOOTH_TYPES[tid]
            is_upper = i < 14
            if is_upper:
                ideal = IDEAL_INCLINATION_UPPER.get(tooth_type, 0.0)
            else:
                ideal = IDEAL_INCLINATION_LOWER.get(tooth_type, 0.0)
            roll, _, _ = _quaternion_to_euler(config[i, :4])
            deviation = abs(roll - ideal)
            deviations.append(deviation)

        mean_deviation = np.mean(deviations)
        return max(0.0, 1.0 - mean_deviation / 20.0)

    def score_rotations(self, config: np.ndarray) -> float:
        """
        Andrews Key 4: No rotations around the long axis.
        Yaw angle should be near zero for all teeth.
        """
        yaw_deviations = []
        for i in range(N_TEETH):
            _, _, yaw = _quaternion_to_euler(config[i, :4])
            yaw_deviations.append(abs(yaw))

        mean_yaw = np.mean(yaw_deviations)
        return max(0.0, 1.0 - mean_yaw / 15.0)

    def score_contact_tightness(self, config: np.ndarray) -> float:
        """
        Andrews Key 5: Tight contacts (no spacing between adjacent teeth).
        Approximate inter-tooth gap from centroid distances and average crown widths.
        """
        gap_penalties = []
        for (a, b) in ARCH_ADJACENCY:
            idx_a = TOOTH_IDS.index(a)
            idx_b = TOOTH_IDS.index(b)
            centroid_dist = float(np.linalg.norm(config[idx_a, 4:7] - config[idx_b, 4:7]))
            expected_dist = (CROWN_WIDTHS[TOOTH_TYPES[a]] + CROWN_WIDTHS[TOOTH_TYPES[b]]) / 2
            gap = abs(centroid_dist - expected_dist)
            gap_penalties.append(min(gap / 3.0, 1.0))

        return max(0.0, 1.0 - np.mean(gap_penalties))

    def score_curve_of_spee(self, config: np.ndarray) -> float:
        """
        Andrews Key 6: Flat curve of Spee.
        Measure depth of the lower arch occlusal plane curvature.
        Ideal: 0-2.5mm depth.
        """
        # Lower arch z-positions: indices 14-27
        lower_tz = config[14:28, 6]

        # Fit a line through endpoints, measure max deviation
        n = len(lower_tz)
        if n < 3:
            return 1.0
        x = np.arange(n, dtype=np.float64)
        slope = (lower_tz[-1] - lower_tz[0]) / (n - 1)
        line = lower_tz[0] + slope * x
        deviations = np.abs(lower_tz - line)
        max_depth = float(deviations.max())

        if max_depth <= 2.5:
            return 1.0
        else:
            return max(0.0, 1.0 - (max_depth - 2.5) / 3.0)

    def score_arch_symmetry(self, config: np.ndarray) -> float:
        """
        Left-right arch symmetry.
        Compare positions of corresponding left/right teeth.
        """
        asymmetries = []
        # Upper: right (0-6) vs left (7-13), mirrored on x-axis
        for i in range(7):
            right = config[i, 4:7].copy()
            left = config[7 + i, 4:7].copy()
            # Mirror left x-position
            left[0] = -left[0]
            asymmetry = float(np.linalg.norm(right - left))
            asymmetries.append(asymmetry)

        # Lower: right (21-27) vs left (14-20)
        for i in range(7):
            right = config[21 + i, 4:7].copy()
            left = config[14 + i, 4:7].copy()
            left[0] = -left[0]
            asymmetry = float(np.linalg.norm(right - left))
            asymmetries.append(asymmetry)

        mean_asymmetry = np.mean(asymmetries)
        return max(0.0, 1.0 - mean_asymmetry / 5.0)
