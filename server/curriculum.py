"""
Adaptive difficulty curriculum controller.
Tracks per-axis mastery and auto-escalates difficulty when the agent
scores consistently above a threshold.
"""
from typing import Dict, List, Any
from .dental_constants import ADAPTIVE_DIFFICULTY_DEFAULTS, ADAPTIVE_DIFFICULTY_RANGES


class CurriculumController:
    """
    Manages adaptive difficulty progression across episodes.

    Tracks reward history and increases difficulty on axes where the
    agent has mastered the current level (>threshold for N consecutive
    episodes). Difficulty only increases, never decreases.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        consecutive_required: int = 3,
        step_size: float = 0.15,
    ):
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self.step_size = step_size  # fraction of range to advance per escalation

        # Current difficulty parameters (start at defaults)
        self.params: Dict[str, Any] = dict(ADAPTIVE_DIFFICULTY_DEFAULTS)

        # Reward history for escalation decisions
        self.episode_rewards: List[float] = []

    def record_episode(self, terminal_reward: float) -> None:
        """Record a completed episode's terminal reward."""
        self.episode_rewards.append(terminal_reward)

    def should_escalate(self) -> bool:
        """Check if the agent has mastered the current difficulty."""
        if len(self.episode_rewards) < self.consecutive_required:
            return False
        recent = self.episode_rewards[-self.consecutive_required:]
        return all(r > self.threshold for r in recent)

    def escalate(self) -> Dict[str, Any]:
        """
        Increase difficulty on the weakest axis (the one closest to its
        current value, i.e., the easiest to escalate).
        Returns updated params.
        """
        if not self.should_escalate():
            return self.params

        # Find which axis has the most room to increase
        best_axis = None
        best_headroom = -1.0

        for axis, spec in ADAPTIVE_DIFFICULTY_RANGES.items():
            if spec['type'] == 'bool':
                continue  # skip boolean axes

            current = self.params[axis]
            max_val = spec['max']
            min_val = spec['min']
            total_range = max_val - min_val

            if spec['type'] == 'int':
                headroom = (max_val - current) / max(total_range, 1)
            else:
                headroom = (max_val - current) / max(total_range, 1e-6)

            if headroom > best_headroom:
                best_headroom = headroom
                best_axis = axis

        if best_axis is None or best_headroom <= 0.01:
            return self.params  # maxed out

        spec = ADAPTIVE_DIFFICULTY_RANGES[best_axis]
        total_range = spec['max'] - spec['min']
        step = total_range * self.step_size

        if spec['type'] == 'int':
            self.params[best_axis] = min(spec['max'], int(self.params[best_axis] + max(1, int(step))))
        else:
            self.params[best_axis] = min(spec['max'], self.params[best_axis] + step)

        # Reset reward history after escalation
        self.episode_rewards.clear()

        return self.params

    def get_params(self) -> Dict[str, Any]:
        """Get current difficulty parameters."""
        return dict(self.params)

    def get_status(self) -> Dict[str, Any]:
        """Get curriculum status for monitoring."""
        return {
            'current_params': dict(self.params),
            'episodes_completed': len(self.episode_rewards),
            'recent_rewards': self.episode_rewards[-self.consecutive_required:] if self.episode_rewards else [],
            'ready_to_escalate': self.should_escalate(),
            'threshold': self.threshold,
            'consecutive_required': self.consecutive_required,
        }
