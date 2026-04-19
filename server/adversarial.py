"""
Adversarial patient non-compliance simulator.

Models real-world treatment disruptions that occur when patients don't
follow the prescribed aligner wear schedule. The RL agent must learn
to detect and recover from these disruptions.

Non-compliance types:
  1. Missed wear: teeth partially revert toward pre-treatment positions
  2. Broken attachment: single tooth loses aligner grip, resets to initial pose
  3. Partial wear: reduced aligner efficacy, movements achieve only 50% of planned

Clinical context:
  - Recommended wear: 22 hrs/day
  - Typical compliance: 60-80% of patients are fully compliant
  - Non-compliance causes treatment delays, refinement aligners, or treatment failure

References:
  - Al-Jamal G, et al. "Clear Aligner Compliance: A Systematic Review."
    Angle Orthod, 93(5):593-601, 2023. DOI: 10.2319/010423-6.1
"""
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .dental_constants import TOOTH_IDS, TOOTH_TYPES, N_TEETH
from .quaternion_utils import (
    quaternion_slerp, quaternion_normalize,
    quaternion_multiply, quaternion_inverse,
    quaternion_from_axis_angle,
)


# Non-compliance event types with clinical descriptions
NONCOMPLIANCE_TYPES = {
    'missed_wear': {
        'description': 'Patient missed aligner wear for 2+ days. Teeth drift back toward pre-treatment positions.',
        'severity': 'moderate',
        'n_teeth_affected': (4, 12),
        'reversion_fraction': (0.2, 0.5),  # 20-50% reversal of last 2 stages
    },
    'broken_attachment': {
        'description': 'Aligner attachment debonded from one tooth. That tooth loses all planned movement.',
        'severity': 'mild',
        'n_teeth_affected': (1, 2),
        'reset_to_initial': True,
    },
    'partial_wear': {
        'description': 'Patient wore aligner only 12-16 hrs/day instead of 22. All movements at 50% efficacy.',
        'severity': 'mild',
        'n_teeth_affected': (28, 28),  # affects all teeth
        'efficacy_fraction': (0.4, 0.6),
    },
}


class AdversarialNonCompliance:
    """
    Simulates patient non-compliance events during aligner treatment.

    Can be used in two modes:
    1. Stochastic: events occur randomly based on per-stage probability
    2. Deterministic: events occur at specified stages (for reproducibility)
    """

    def __init__(
        self,
        event_probability: float = 0.0,
        allowed_types: Optional[List[str]] = None,
        max_events_per_episode: int = 3,
    ):
        """
        Args:
            event_probability: Per-stage probability of a non-compliance event (0.0-1.0)
            allowed_types: List of event types to sample from. None = all types.
            max_events_per_episode: Maximum events in one episode.
        """
        self.event_probability = event_probability
        self.allowed_types = allowed_types or list(NONCOMPLIANCE_TYPES.keys())
        self.max_events = max_events_per_episode
        self.events_triggered = 0
        self.event_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset event counter for a new episode."""
        self.events_triggered = 0
        self.event_log = []

    def maybe_trigger(
        self,
        trajectory: np.ndarray,
        current_stage: int,
        initial_config: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Possibly trigger a non-compliance event at the current stage.

        Args:
            trajectory: (26, 28, 7) trajectory buffer
            current_stage: current stage index (1-24)
            initial_config: (28, 7) pre-treatment configuration
            rng: seeded random generator

        Returns:
            (modified_trajectory, event_info_or_None)
        """
        if self.events_triggered >= self.max_events:
            return trajectory, None

        if rng.random() > self.event_probability:
            return trajectory, None

        if current_stage < 2:
            return trajectory, None  # too early for meaningful disruption

        # Pick event type
        event_type = rng.choice(self.allowed_types)
        spec = NONCOMPLIANCE_TYPES[event_type]

        modified = trajectory.copy()
        event_info = {
            'type': event_type,
            'stage': current_stage,
            'description': spec['description'],
            'severity': spec['severity'],
            'teeth_affected': [],
        }

        if event_type == 'missed_wear':
            modified, affected = self._apply_missed_wear(
                modified, current_stage, initial_config, spec, rng
            )
            event_info['teeth_affected'] = affected

        elif event_type == 'broken_attachment':
            modified, affected = self._apply_broken_attachment(
                modified, current_stage, initial_config, spec, rng
            )
            event_info['teeth_affected'] = affected

        elif event_type == 'partial_wear':
            modified, affected = self._apply_partial_wear(
                modified, current_stage, spec, rng
            )
            event_info['teeth_affected'] = affected

        self.events_triggered += 1
        self.event_log.append(event_info)

        return modified, event_info

    def _apply_missed_wear(
        self,
        trajectory: np.ndarray,
        stage: int,
        initial: np.ndarray,
        spec: dict,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[int]]:
        """Teeth partially revert toward initial positions."""
        n_affect = int(rng.integers(spec['n_teeth_affected'][0], spec['n_teeth_affected'][1] + 1))
        reversion = rng.uniform(spec['reversion_fraction'][0], spec['reversion_fraction'][1])

        indices = rng.choice(N_TEETH, size=min(n_affect, N_TEETH), replace=False)
        affected_teeth = [TOOTH_IDS[i] for i in indices]

        for idx in indices:
            current = trajectory[stage, idx]
            init = initial[idx]
            # SLERP back toward initial by reversion fraction
            trajectory[stage, idx, :4] = quaternion_normalize(
                quaternion_slerp(current[:4], init[:4], reversion)
            )
            trajectory[stage, idx, 4:7] = (1 - reversion) * current[4:7] + reversion * init[4:7]

        return trajectory, affected_teeth

    def _apply_broken_attachment(
        self,
        trajectory: np.ndarray,
        stage: int,
        initial: np.ndarray,
        spec: dict,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[int]]:
        """One tooth resets completely to initial position."""
        n_affect = int(rng.integers(spec['n_teeth_affected'][0], spec['n_teeth_affected'][1] + 1))
        indices = rng.choice(N_TEETH, size=min(n_affect, N_TEETH), replace=False)
        affected_teeth = [TOOTH_IDS[i] for i in indices]

        for idx in indices:
            trajectory[stage, idx] = initial[idx].copy()

        return trajectory, affected_teeth

    def _apply_partial_wear(
        self,
        trajectory: np.ndarray,
        stage: int,
        spec: dict,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[int]]:
        """All teeth only achieve partial movement (50% efficacy)."""
        efficacy = rng.uniform(spec['efficacy_fraction'][0], spec['efficacy_fraction'][1])

        prev = trajectory[stage - 1]
        curr = trajectory[stage]

        for idx in range(N_TEETH):
            # Reduce the delta by efficacy factor
            planned_delta_t = curr[idx, 4:7] - prev[idx, 4:7]
            trajectory[stage, idx, 4:7] = prev[idx, 4:7] + planned_delta_t * efficacy

            # Reduce rotation delta
            q_delta = quaternion_multiply(curr[idx, :4], quaternion_inverse(prev[idx, :4]))
            # Interpolate q_delta toward identity by (1-efficacy)
            identity = np.array([1.0, 0.0, 0.0, 0.0])
            q_partial = quaternion_normalize(quaternion_slerp(identity, q_delta, efficacy))
            trajectory[stage, idx, :4] = quaternion_normalize(
                quaternion_multiply(q_partial, prev[idx, :4])
            )

        return trajectory, [TOOTH_IDS[i] for i in range(N_TEETH)]

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Return log of all events triggered in this episode."""
        return self.event_log

    def get_status(self) -> Dict[str, Any]:
        """Return current adversarial status."""
        return {
            'event_probability': self.event_probability,
            'events_triggered': self.events_triggered,
            'max_events': self.max_events,
            'allowed_types': self.allowed_types,
            'event_log': self.event_log,
        }


def list_noncompliance_types() -> Dict[str, Dict[str, str]]:
    """Return metadata about available non-compliance event types."""
    return {k: {'description': v['description'], 'severity': v['severity']}
            for k, v in NONCOMPLIANCE_TYPES.items()}
