"""
Clinically classified malocclusion generation and parametric arch shape variation.

Instead of random perturbations, generates malocclusion patterns that match
real clinical classifications (Angle Class I/II/III, deep bite, open bite,
crossbite, crowding, spacing).

Also provides parametric arch curve generation using clinically validated
mathematical models (beta function, catenary, polynomial).

References:
  - Angle EH. Classification of malocclusion. Dental Cosmos, 41:248-264, 1899.
  - Braun S et al. Form of the human dental arch. Angle Orthod, 68(1):29-36, 1998.
  - Noroozi H et al. The dental arch form revisited. Angle Orthod, 71(5):386-389, 2001.
  - AlHarbi S et al. Mathematical analyses of dental arch curvatures. Angle Orthod, 78(2):281-287, 2008.
"""
import math
import numpy as np
from typing import Dict, Any, Tuple

from .dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH,
    IDEAL_UPPER_TX, IDEAL_UPPER_TY, IDEAL_UPPER_TZ,
    IDEAL_LOWER_TX, IDEAL_LOWER_TY, IDEAL_LOWER_TZ,
)
from .quaternion_utils import (
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_normalize,
)


# ---------------------------------------------------------------------------
# Malocclusion classification patterns
# ---------------------------------------------------------------------------

MALOCCLUSION_CLASSES = {
    'class_I_crowding': {
        'description': 'Teeth displaced within arch, normal molar relationship',
        'difficulty': 'easy',
        'n_teeth': (6, 12),
        'trans_range': (1.0, 3.5),
        'rot_range': (5.0, 15.0),
        'pattern': 'crowding',
    },
    'class_I_spacing': {
        'description': 'Gaps between teeth, normal molar relationship',
        'difficulty': 'easy',
        'n_teeth': (4, 8),
        'trans_range': (1.0, 3.0),
        'rot_range': (3.0, 10.0),
        'pattern': 'spacing',
    },
    'class_II_div1': {
        'description': 'Upper protrusion, increased overjet (>5mm)',
        'difficulty': 'medium',
        'n_teeth': (10, 16),
        'trans_range': (2.0, 5.0),
        'rot_range': (8.0, 20.0),
        'pattern': 'class_II_div1',
    },
    'class_II_div2': {
        'description': 'Upper incisors retroclined, deep bite',
        'difficulty': 'medium',
        'n_teeth': (8, 14),
        'trans_range': (2.0, 4.5),
        'rot_range': (10.0, 25.0),
        'pattern': 'class_II_div2',
    },
    'class_III': {
        'description': 'Lower protrusion, negative overjet',
        'difficulty': 'hard',
        'n_teeth': (12, 20),
        'trans_range': (3.0, 6.0),
        'rot_range': (10.0, 25.0),
        'pattern': 'class_III',
    },
    'open_bite': {
        'description': 'No vertical overlap of incisors',
        'difficulty': 'hard',
        'n_teeth': (8, 14),
        'trans_range': (2.0, 5.0),
        'rot_range': (10.0, 20.0),
        'pattern': 'open_bite',
    },
    'crossbite': {
        'description': 'Transverse discrepancy, upper teeth inside lower',
        'difficulty': 'hard',
        'n_teeth': (6, 12),
        'trans_range': (2.0, 5.0),
        'rot_range': (8.0, 20.0),
        'pattern': 'crossbite',
    },
    'asymmetric': {
        'description': 'Different classification left vs right',
        'difficulty': 'very_hard',
        'n_teeth': (14, 22),
        'trans_range': (3.0, 7.0),
        'rot_range': (10.0, 30.0),
        'pattern': 'asymmetric',
    },
}

# Incisor indices (upper + lower)
_UPPER_INCISORS = [0, 1, 7, 8]      # FDI 11,12,21,22
_LOWER_INCISORS = [14, 15, 21, 22]   # FDI 31,32,41,42
_UPPER_CANINES = [2, 9]              # FDI 13, 23
_LOWER_CANINES = [16, 23]            # FDI 33, 43
_UPPER_MOLARS = [5, 6, 12, 13]       # FDI 16,17,26,27
_LOWER_MOLARS = [19, 20, 26, 27]     # FDI 36,37,46,47
_UPPER_PREMOLARS = [3, 4, 10, 11]    # FDI 14,15,24,25
_LOWER_PREMOLARS = [17, 18, 24, 25]  # FDI 34,35,44,45
_LEFT_SIDE = list(range(7, 14)) + list(range(14, 21))   # upper left + lower left
_RIGHT_SIDE = list(range(0, 7)) + list(range(21, 28))    # upper right + lower right


def apply_classified_malocclusion(
    ideal: np.ndarray,
    malocclusion_class: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply a clinically classified malocclusion pattern to the ideal arch.

    Args:
        ideal: (28, 7) ideal configuration
        malocclusion_class: Key from MALOCCLUSION_CLASSES
        rng: seeded random generator

    Returns:
        (28, 7) perturbed configuration matching the clinical pattern
    """
    config = ideal.copy()
    spec = MALOCCLUSION_CLASSES.get(malocclusion_class)
    if spec is None:
        return config

    n_teeth = int(rng.integers(spec['n_teeth'][0], spec['n_teeth'][1] + 1))
    trans_lo, trans_hi = spec['trans_range']
    rot_lo, rot_hi = spec['rot_range']
    pattern = spec['pattern']

    if pattern == 'crowding':
        # Crowding: adjacent teeth displaced toward each other, rotated
        # Primarily affects anterior teeth
        targets = _UPPER_INCISORS + _LOWER_INCISORS + _UPPER_CANINES + _LOWER_CANINES
        indices = rng.choice(targets, size=min(n_teeth, len(targets)), replace=False)
        for idx in indices:
            # Displace mesially/distally (along arch)
            mag = rng.uniform(trans_lo, trans_hi)
            config[idx, 4] += rng.uniform(-1, 1) * mag * 0.5  # lateral
            config[idx, 5] += rng.uniform(-1, 1) * mag * 0.5  # AP
            # Rotate around long axis (crowding rotation)
            rot = rng.uniform(rot_lo, rot_hi)
            axis = np.array([0.0, 0.0, 1.0])
            dq = quaternion_from_axis_angle(axis, math.radians(rot * rng.choice([-1, 1])))
            config[idx, :4] = quaternion_normalize(quaternion_multiply(dq, config[idx, :4]))

    elif pattern == 'spacing':
        # Spacing: teeth spread apart, gaps between them
        targets = _UPPER_INCISORS + _LOWER_INCISORS
        indices = rng.choice(targets, size=min(n_teeth, len(targets)), replace=False)
        for idx in indices:
            # Displace outward from center
            direction = np.sign(config[idx, 4]) if abs(config[idx, 4]) > 0.1 else rng.choice([-1, 1])
            mag = rng.uniform(trans_lo, trans_hi)
            config[idx, 4] += direction * mag  # lateral spacing

    elif pattern == 'class_II_div1':
        # Class II div 1: upper incisors protruding forward (increased overjet)
        for idx in _UPPER_INCISORS:
            config[idx, 5] -= rng.uniform(2.0, 5.0)  # push upper incisors forward (reduce Y)
            # Procline upper incisors (tip labially)
            rot = rng.uniform(5.0, 15.0)
            axis = np.array([1.0, 0.0, 0.0])
            dq = quaternion_from_axis_angle(axis, math.radians(rot))
            config[idx, :4] = quaternion_normalize(quaternion_multiply(dq, config[idx, :4]))
        # Also perturb some random teeth
        extra = rng.choice(range(N_TEETH), size=min(n_teeth - 4, N_TEETH), replace=False)
        for idx in extra:
            mag = rng.uniform(trans_lo * 0.5, trans_hi * 0.5)
            config[idx, 4:7] += rng.standard_normal(3) * mag * 0.3

    elif pattern == 'class_II_div2':
        # Class II div 2: upper incisors retroclined, deep bite
        for idx in _UPPER_INCISORS:
            config[idx, 5] += rng.uniform(1.0, 3.0)  # retract slightly
            # Retrocline (tip lingually)
            rot = rng.uniform(10.0, 25.0)
            axis = np.array([1.0, 0.0, 0.0])
            dq = quaternion_from_axis_angle(axis, math.radians(-rot))
            config[idx, :4] = quaternion_normalize(quaternion_multiply(dq, config[idx, :4]))
        # Deep bite: lower incisors move upward
        for idx in _LOWER_INCISORS:
            config[idx, 6] += rng.uniform(1.0, 3.0)  # increase overbite

    elif pattern == 'class_III':
        # Class III: lower jaw protrusion (negative overjet)
        for idx in _LOWER_INCISORS:
            config[idx, 5] -= rng.uniform(2.0, 5.0)  # push lower incisors forward
        for idx in _LOWER_CANINES:
            config[idx, 5] -= rng.uniform(1.0, 3.0)
        # Upper incisors may also be retruded
        for idx in _UPPER_INCISORS:
            config[idx, 5] += rng.uniform(0.5, 2.0)

    elif pattern == 'open_bite':
        # Open bite: anterior teeth don't overlap vertically
        for idx in _UPPER_INCISORS + _UPPER_CANINES:
            config[idx, 6] -= rng.uniform(2.0, 4.0)  # upper teeth move up
        for idx in _LOWER_INCISORS + _LOWER_CANINES:
            config[idx, 6] += rng.uniform(1.0, 3.0)  # lower teeth move down

    elif pattern == 'crossbite':
        # Crossbite: one side has reversed buccal-lingual relationship
        # Posterior crossbite on one side
        side = rng.choice([_LEFT_SIDE, _RIGHT_SIDE])
        for idx in side:
            if idx in _UPPER_PREMOLARS + _UPPER_MOLARS:
                config[idx, 4] -= rng.uniform(2.0, 4.0)  # narrow upper
            elif idx in _LOWER_PREMOLARS + _LOWER_MOLARS:
                config[idx, 4] += rng.uniform(1.0, 2.0)  # widen lower

    elif pattern == 'asymmetric':
        # Different pattern on each side
        # Left side: Class II pattern
        for idx in [i for i in _LEFT_SIDE if i in _UPPER_INCISORS]:
            config[idx, 5] -= rng.uniform(2.0, 4.0)
        # Right side: crowding pattern
        right_anterior = [i for i in _RIGHT_SIDE if i < 7 or i >= 21]
        for idx in rng.choice(right_anterior, size=min(4, len(right_anterior)), replace=False):
            mag = rng.uniform(trans_lo, trans_hi)
            config[idx, 4:7] += rng.standard_normal(3) * mag * 0.5
            rot = rng.uniform(rot_lo, rot_hi)
            axis = rng.standard_normal(3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            dq = quaternion_from_axis_angle(axis, math.radians(rot))
            config[idx, :4] = quaternion_normalize(quaternion_multiply(dq, config[idx, :4]))

    return config


# ---------------------------------------------------------------------------
# Parametric arch shape generation
# ---------------------------------------------------------------------------

ARCH_FORMS = ['ovoid', 'tapered', 'square', 'catenary', 'beta']


def generate_arch_positions(
    arch_form: str = 'ovoid',
    arch_width_mm: float = 70.0,
    arch_depth_mm: float = 38.0,
    n_teeth_per_side: int = 7,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tooth positions along a parametric arch curve.

    Args:
        arch_form: One of 'ovoid', 'tapered', 'square', 'catenary', 'beta'
        arch_width_mm: Total inter-molar width
        arch_depth_mm: Anterior-posterior depth (incisor to molar line)
        n_teeth_per_side: Teeth per quadrant (default 7: I1,I2,C,PM1,PM2,M1,M2)

    Returns:
        (tx, ty) arrays each of length 2*n_teeth_per_side (right side + left side)
    """
    if rng is not None:
        # Add natural variation
        arch_width_mm += rng.normal(0, 3.0)
        arch_depth_mm += rng.normal(0, 2.0)

    half_width = arch_width_mm / 2.0
    n = n_teeth_per_side

    # Parameter t: 0 = right molar, 0.5 = central incisor, 1.0 = left molar
    t_right = np.linspace(0.0, 0.5, n)  # right side (molar → incisor)
    t_left = np.linspace(0.5, 1.0, n)   # left side (incisor → molar)

    if arch_form == 'catenary':
        # y = a * cosh(x/a) - a
        a = arch_depth_mm * 0.8
        x_right = np.linspace(half_width, 0, n)
        x_left = np.linspace(0, -half_width, n)
        y_right = a * (np.cosh(x_right / max(a, 1)) - 1)
        y_left = a * (np.cosh(x_left / max(a, 1)) - 1)

    elif arch_form == 'beta':
        # Y = A*X^6 + B*X^2 (beta function approximation)
        A = arch_depth_mm / (half_width**6 + half_width**2 + 1e-6) * 0.001
        B = arch_depth_mm / (half_width**2 + 1e-6) * 0.9
        x_right = np.linspace(half_width, 0, n)
        x_left = np.linspace(0, -half_width, n)
        y_right = A * x_right**6 + B * x_right**2
        y_left = A * x_left**6 + B * x_left**2

    elif arch_form == 'tapered':
        # V-shaped, narrower anterior
        x_right = np.linspace(half_width, 0, n)
        x_left = np.linspace(0, -half_width, n)
        # Parabolic with steeper taper
        y_right = arch_depth_mm * (1 - (x_right / half_width)**1.5)
        y_left = arch_depth_mm * (1 - (np.abs(x_left) / half_width)**1.5)

    elif arch_form == 'square':
        # U-shaped, flatter anterior
        x_right = np.linspace(half_width, 0, n)
        x_left = np.linspace(0, -half_width, n)
        y_right = arch_depth_mm * (1 - (x_right / half_width)**3)
        y_left = arch_depth_mm * (1 - (np.abs(x_left) / half_width)**3)

    else:  # 'ovoid' (default)
        # Standard parabolic
        x_right = np.linspace(half_width, 0, n)
        x_left = np.linspace(0, -half_width, n)
        y_right = arch_depth_mm * (1 - (x_right / half_width)**2)
        y_left = arch_depth_mm * (1 - (np.abs(x_left) / half_width)**2)

    tx = np.concatenate([x_right, x_left])
    ty = np.concatenate([y_right, y_left])

    return tx, ty


def generate_config_with_arch_form(
    arch_form: str = 'ovoid',
    arch_width_mm: float = 70.0,
    arch_depth_mm: float = 38.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a (28, 7) ideal configuration using a parametric arch form.

    Returns config with identity quaternions and positions from the arch curve.
    """
    config = np.zeros((N_TEETH, 7), dtype=np.float64)
    config[:, 0] = 1.0  # identity quaternion

    # Upper arch
    tx_upper, ty_upper = generate_arch_positions(
        arch_form, arch_width_mm, arch_depth_mm, 7, rng
    )
    for i in range(14):
        config[i, 4] = tx_upper[i]
        config[i, 5] = ty_upper[i]
        config[i, 6] = 0.0

    # Lower arch (slightly narrower, offset down)
    tx_lower, ty_lower = generate_arch_positions(
        arch_form, arch_width_mm * 0.95, arch_depth_mm * 0.95, 7, rng
    )
    for i in range(14):
        config[14 + i, 4] = tx_lower[i]
        config[14 + i, 5] = ty_lower[i]
        config[14 + i, 6] = -2.0  # lower arch offset

    return config


def list_malocclusion_classes() -> Dict[str, Dict[str, Any]]:
    """Return metadata about available malocclusion classes."""
    return {k: {'description': v['description'], 'difficulty': v['difficulty']}
            for k, v in MALOCCLUSION_CLASSES.items()}


def list_arch_forms() -> list:
    """Return available arch form types."""
    return ARCH_FORMS
