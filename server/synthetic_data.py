"""
Synthetic dental case generator for the aligner trajectory planner.
Produces deterministic, seeded dental malocclusion cases.
"""
import math
import numpy as np
from typing import Dict, Any

from typing import List, Optional
from .dental_constants import (
    TOOTH_IDS, N_TEETH, N_STAGES,
    TOOTH_TYPES, STAGING_PRIORITY,
    IDEAL_UPPER_TX, IDEAL_UPPER_TY, IDEAL_UPPER_TZ,
    IDEAL_LOWER_TX, IDEAL_LOWER_TY, IDEAL_LOWER_TZ,
    ADAPTIVE_DIFFICULTY_DEFAULTS, ADAPTIVE_DIFFICULTY_RANGES,
)
from .quaternion_utils import (
    quaternion_slerp,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_normalize,
    quaternion_from_axis_angle,
    random_quaternion_perturbation,
)


class DentalCaseGenerator:
    """
    Generates synthetic dental cases with malocclusion perturbations.
    All generation is seeded for reproducibility.
    """

    def generate_ideal_config(self) -> np.ndarray:
        """
        Build the 28x7 ideal dental configuration.
        All rotations are identity quaternion (1,0,0,0).
        """
        config = np.zeros((N_TEETH, 7), dtype=np.float64)
        config[:, 0] = 1.0  # qw = 1 (identity rotation)

        # Upper arch: first 14 teeth (indices 0-13) = tooth IDs 11-17, 21-27
        for i in range(14):
            config[i, 4] = IDEAL_UPPER_TX[i]
            config[i, 5] = IDEAL_UPPER_TY[i]
            config[i, 6] = IDEAL_UPPER_TZ[i]

        # Lower arch: last 14 teeth (indices 14-27) = tooth IDs 31-37, 41-47
        for i in range(14):
            config[14 + i, 4] = IDEAL_LOWER_TX[i]
            config[14 + i, 5] = IDEAL_LOWER_TY[i]
            config[14 + i, 6] = IDEAL_LOWER_TZ[i]

        return config

    def apply_malocclusion(
        self,
        ideal: np.ndarray,
        difficulty: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Perturb the ideal config to generate a malocclusion case.

        difficulty:
          'easy'   : 4-6 teeth, 1-3mm translation, 5-15 deg rotation (z-axis tipping)
          'medium' : 10-14 teeth, 2-5mm, 10-20 deg (multi-axis)
          'hard'   : 18-24 teeth, 3-8mm, 15-25 deg (combined motions)
        """
        config = ideal.copy()

        if difficulty == 'easy':
            n_perturb = int(rng.integers(4, 7))
            trans_range = (1.0, 3.0)
            rot_range = (5.0, 15.0)
            axes = ['z']
        elif difficulty == 'medium':
            n_perturb = int(rng.integers(10, 15))
            trans_range = (2.0, 5.0)
            rot_range = (10.0, 20.0)
            axes = ['x', 'y', 'z']
        else:  # hard
            n_perturb = int(rng.integers(18, 25))
            trans_range = (3.0, 8.0)
            rot_range = (15.0, 25.0)
            axes = ['x', 'y', 'z']

        indices = rng.choice(N_TEETH, size=n_perturb, replace=False)

        for idx in indices:
            # Apply random translation
            trans_mag = rng.uniform(*trans_range)
            direction = rng.standard_normal(3)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if difficulty == 'easy':
                direction[2] = 0.0  # mostly in-plane for easy
                direction /= (np.linalg.norm(direction) + 1e-12)
            config[idx, 4:7] += direction * trans_mag

            # Apply random rotation
            rot_deg = rng.uniform(*rot_range)
            if 'z' in axes and difficulty == 'easy':
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = rng.standard_normal(3)
                axis /= (np.linalg.norm(axis) + 1e-12)

            delta_q = quaternion_from_axis_angle(axis, math.radians(rot_deg))
            old_q = config[idx, :4]
            new_q = quaternion_normalize(quaternion_multiply(delta_q, old_q))
            config[idx, :4] = new_q

        return config

    def generate_baseline_trajectory(
        self,
        initial: np.ndarray,
        final: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a 26x28x7 SLERP baseline trajectory.
        stage 0 = initial, stage 25 = final, stages 1-24 = interpolated.
        """
        trajectory = np.zeros((26, N_TEETH, 7), dtype=np.float64)
        trajectory[0] = initial.copy()
        trajectory[25] = final.copy()

        for k in range(1, 25):
            alpha = k / 25.0
            for i in range(N_TEETH):
                # SLERP rotation
                q0 = initial[i, :4]
                q1 = final[i, :4]
                trajectory[k, i, :4] = quaternion_slerp(q0, q1, alpha)

                # Linear interpolation for translation
                t0 = initial[i, 4:7]
                t1 = final[i, 4:7]
                trajectory[k, i, 4:7] = (1.0 - alpha) * t0 + alpha * t1

        return trajectory

    def compute_delta_poses(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Given 26x28x7 trajectory, compute 25x28x7 delta poses between consecutive stages.
        delta[k][i] = relative pose from stage k to k+1 for tooth i.
        """
        n_steps = trajectory.shape[0] - 1
        deltas = np.zeros((n_steps, N_TEETH, 7), dtype=np.float64)

        for k in range(n_steps):
            for i in range(N_TEETH):
                # Delta translation
                deltas[k, i, 4:7] = trajectory[k+1, i, 4:7] - trajectory[k, i, 4:7]

                # Delta rotation: q_delta = q_{k+1} * q_k^{-1}
                q_k = trajectory[k, i, :4]
                q_k1 = trajectory[k+1, i, :4]
                q_delta = quaternion_normalize(
                    quaternion_multiply(q_k1, quaternion_inverse(q_k))
                )
                deltas[k, i, :4] = q_delta

        return deltas

    def generate_case(self, difficulty: str, seed: int) -> Dict[str, Any]:
        """
        Generate a complete dental case dict.
        Same seed -> same case (reproducibility requirement).
        """
        rng = np.random.default_rng(seed)
        ideal = self.generate_ideal_config()
        initial = self.apply_malocclusion(ideal, difficulty, rng)
        baseline_traj = self.generate_baseline_trajectory(initial, ideal)

        return {
            'initial_config':      initial,        # shape (28, 7)
            'target_config':       ideal,           # shape (28, 7)
            'tooth_ids':           TOOTH_IDS,       # list of 28 FDI IDs
            'tooth_types':         TOOTH_TYPES,
            'baseline_trajectory': baseline_traj,  # shape (26, 28, 7)
            'difficulty':          difficulty,
            'seed':                seed,
        }

    def generate_case_from_dataset(
        self,
        source: str,
        patient_path: str,
        difficulty: str = 'medium',
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Generate a dental case using real clinical data as the target config.
        The real data defines the aligned (target) tooth poses;
        malocclusion is synthetically applied to create the initial config.

        Args:
            source: Dataset source name ('open_full_jaw', 'teeth3ds', 'mendeley_jaw')
            patient_path: Path to the patient data (directory or file)
            difficulty: Difficulty level for synthetic malocclusion
            seed: Random seed for reproducibility

        Returns:
            Same dict format as generate_case() but with real anatomy.
        """
        from .dataset_loader import (
            load_open_full_jaw, load_teeth3ds, load_mendeley_jaw,
        )

        # Load real target config from dataset
        if source == 'open_full_jaw':
            target = load_open_full_jaw(patient_path)
        elif source == 'teeth3ds':
            # patient_path should be "obj_path:json_path"
            parts = patient_path.split(':')
            target = load_teeth3ds(parts[0], parts[1])
        elif source == 'mendeley_jaw':
            target = load_mendeley_jaw(patient_path)
        else:
            raise ValueError(f"Unknown dataset source: {source}")

        # Fill in any missing teeth (zero translation) with ideal positions
        ideal = self.generate_ideal_config()
        for i in range(N_TEETH):
            if np.abs(target[i, 4:7]).sum() < 1e-6:
                target[i] = ideal[i]

        # Apply synthetic malocclusion to create initial config
        rng = np.random.default_rng(seed)
        initial = self.apply_malocclusion(target.copy(), difficulty, rng)
        baseline_traj = self.generate_baseline_trajectory(initial, target)

        return {
            'initial_config':      initial,
            'target_config':       target,
            'tooth_ids':           TOOTH_IDS,
            'tooth_types':         TOOTH_TYPES,
            'baseline_trajectory': baseline_traj,
            'difficulty':          difficulty,
            'seed':                seed,
            'data_source':         source,
            'patient_path':        patient_path,
        }

    def apply_malocclusion_adaptive(
        self,
        ideal: np.ndarray,
        params: Dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Perturb ideal config using continuous adaptive difficulty parameters.

        params keys (all optional, defaults from ADAPTIVE_DIFFICULTY_DEFAULTS):
          n_perturbed_teeth:     int   4-28
          translation_magnitude: float 0.5-8.0 mm
          rotation_magnitude:    float 5.0-35.0 degrees
          multi_axis_rotation:   bool  single vs multi-axis
          missing_teeth:         int   0-4 teeth set to identity (no movement)
        """
        config = ideal.copy()

        n_perturb = int(params.get('n_perturbed_teeth', ADAPTIVE_DIFFICULTY_DEFAULTS['n_perturbed_teeth']))
        trans_mag_max = float(params.get('translation_magnitude', ADAPTIVE_DIFFICULTY_DEFAULTS['translation_magnitude']))
        rot_mag_max = float(params.get('rotation_magnitude', ADAPTIVE_DIFFICULTY_DEFAULTS['rotation_magnitude']))
        multi_axis = bool(params.get('multi_axis_rotation', ADAPTIVE_DIFFICULTY_DEFAULTS['multi_axis_rotation']))
        missing = int(params.get('missing_teeth', ADAPTIVE_DIFFICULTY_DEFAULTS['missing_teeth']))

        # Clamp to valid ranges
        n_perturb = max(1, min(N_TEETH, n_perturb))
        trans_mag_max = max(0.1, min(10.0, trans_mag_max))
        rot_mag_max = max(1.0, min(45.0, rot_mag_max))
        missing = max(0, min(min(missing, N_TEETH - n_perturb), 4))

        indices = rng.choice(N_TEETH, size=n_perturb, replace=False)

        for idx in indices:
            trans_mag = rng.uniform(trans_mag_max * 0.3, trans_mag_max)
            direction = rng.standard_normal(3)
            direction /= (np.linalg.norm(direction) + 1e-12)
            if not multi_axis:
                direction[2] = 0.0
                direction /= (np.linalg.norm(direction) + 1e-12)
            config[idx, 4:7] += direction * trans_mag

            rot_deg = rng.uniform(rot_mag_max * 0.3, rot_mag_max)
            if multi_axis:
                axis = rng.standard_normal(3)
                axis /= (np.linalg.norm(axis) + 1e-12)
            else:
                axis = np.array([0.0, 0.0, 1.0])
            delta_q = quaternion_from_axis_angle(axis, math.radians(rot_deg))
            old_q = config[idx, :4]
            config[idx, :4] = quaternion_normalize(quaternion_multiply(delta_q, old_q))

        # Set missing teeth to identity (simulate missing/extracted teeth)
        if missing > 0:
            non_perturbed = [i for i in range(N_TEETH) if i not in indices]
            if len(non_perturbed) >= missing:
                missing_idx = rng.choice(non_perturbed, size=missing, replace=False)
                for idx in missing_idx:
                    config[idx] = ideal[idx]  # no movement needed

        return config

    def generate_case_adaptive(
        self,
        params: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """
        Generate a case with continuous adaptive difficulty parameters.

        params: dict with keys from ADAPTIVE_DIFFICULTY_RANGES
        seed: random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        ideal = self.generate_ideal_config()
        initial = self.apply_malocclusion_adaptive(ideal, params, rng)
        baseline_traj = self.generate_baseline_trajectory(initial, ideal)

        return {
            'initial_config':      initial,
            'target_config':       ideal,
            'tooth_ids':           TOOTH_IDS,
            'tooth_types':         TOOTH_TYPES,
            'baseline_trajectory': baseline_traj,
            'difficulty':          'adaptive',
            'difficulty_params':   params,
            'seed':                seed,
        }

    def apply_adversarial_jitter(
        self,
        trajectory: np.ndarray,
        current_stage: int,
        jitter_strength: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply SE(3) jitter to trajectory[current_stage].
        jitter_strength: 0.1 = mild (0.1mm, 1deg), 0.3 = strong (0.3mm, 3deg)
        """
        perturbed = trajectory.copy()

        # Select 1-4 teeth to jitter
        n_jitter = int(rng.integers(1, 5))
        tooth_indices = rng.choice(N_TEETH, size=n_jitter, replace=False)

        for idx in tooth_indices:
            # Translation jitter
            trans_noise = rng.standard_normal(3) * jitter_strength
            perturbed[current_stage, idx, 4:7] += trans_noise

            # Rotation jitter
            max_angle_deg = jitter_strength * 10.0  # 0.1 -> 1 deg, 0.3 -> 3 deg
            perturbed[current_stage, idx, :4] = random_quaternion_perturbation(
                perturbed[current_stage, idx, :4],
                max_angle_deg,
                rng,
            )

        return perturbed, tooth_indices.tolist()
