"""
Dental Aligner Trajectory Planning Environment.

Episode structure:
  Step 1: Agent receives initial+target config, submits 24-stage plan.
  [task_hard] Step 2: Adversarial jitter applied, agent revises remaining stages.
  Episode ends after 3 steps max, or after step 1 for easy/medium.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from openenv.core.env_server.interfaces import Environment

from models import AlignerAction, AlignerObservation, AlignerState, ToothPoseTableRow
from server.synthetic_data import DentalCaseGenerator
from server.grader import AlignerGrader
from server.dental_constants import (
    TOOTH_IDS, TOOTH_TYPES, N_TEETH, N_STAGES,
    ARCH_ADJACENCY, STAGING_PRIORITY,
)
from server.quaternion_utils import quaternion_to_angle_deg, quaternion_multiply, quaternion_inverse


# ---------------------------------------------------------------------------
# Task description strings
# ---------------------------------------------------------------------------

TASK_DESCRIPTION_EASY = """DENTAL ALIGNER PLANNING TASK (EASY) — battisiBot
You are an expert orthodontic treatment planning AI.
Given 28 teeth in malocclusion, plan exactly 24 aligner stages that smoothly
move each tooth from its current pose to its target pose.

Each tooth is a 7-vector: [qw, qx, qy, qz, tx, ty, tz]
  qw,qx,qy,qz = unit quaternion rotation (must satisfy qw^2+qx^2+qy^2+qz^2=1)
  tx, ty, tz  = translation in millimetres

CLINICAL CONSTRAINTS (enforced by grader):
  - Max translation per tooth per stage: 0.25 mm
  - Max rotation per tooth per stage: 2.0 degrees
  - All quaternions must be unit quaternions
  - Stage 0=initial (given), Stage 25=target (given). Output stages 1-24.

SCORING: final_accuracy 40% + smoothness 20% + compliance 20% + staging_quality 20%
A naive SLERP baseline scores ~0.40. Beat it by prioritising incisors first.

HINT: Use SLERP interpolation as your baseline. Move incisors (teeth 11-13,21-23,41-43,31-33) earlier, molars (16,17,26,27,36,37,46,47) later.""".strip()

TASK_DESCRIPTION_MEDIUM = TASK_DESCRIPTION_EASY.replace(
    'DENTAL ALIGNER PLANNING TASK (EASY)',
    'DENTAL ALIGNER PLANNING TASK (MEDIUM)',
)

TASK_DESCRIPTION_HARD = (
    TASK_DESCRIPTION_EASY.replace(
        'DENTAL ALIGNER PLANNING TASK (EASY)',
        'DENTAL ALIGNER PLANNING TASK (HARD)',
    )
    + '\n\nADVERSARIAL MODE: After your initial plan, jitter will be injected into one stage\n'
    '(simulating patient non-compliance). You will then revise remaining stages.\n'
    'Recovery quality contributes 15% to your final score.'
)

_TASK_DESCRIPTIONS = {
    'easy':   TASK_DESCRIPTION_EASY,
    'medium': TASK_DESCRIPTION_MEDIUM,
    'hard':   TASK_DESCRIPTION_HARD,
}


# ---------------------------------------------------------------------------
# Module-level session store so state persists across HTTP request cycles
# ---------------------------------------------------------------------------
_SESSIONS: Dict[str, dict] = {}
_LAST_EPISODE_ID: Optional[str] = None


class DentalAlignerEnvironment(Environment):
    """
    OpenEnv environment for dental aligner trajectory planning.

    SUPPORTS_CONCURRENT_SESSIONS = True: session state is stored in
    module-level dict keyed by episode_id.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 3

    def __init__(self):
        super().__init__()
        self._case_gen = DentalCaseGenerator()
        self._grader = AlignerGrader()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AlignerObservation:
        """
        Reset — start a new episode.

        Args:
            seed:       Random seed. Derived from episode_id hash if not provided.
            episode_id: Episode ID. Auto-generated UUID if not provided.
            task_id:    'task_easy', 'task_medium', or 'task_hard'.
                        Defaults to 'task_easy' if not provided.
        """
        global _LAST_EPISODE_ID

        # 1. Generate episode_id if not provided
        if episode_id is None:
            episode_id = str(uuid.uuid4())

        # 2. Set seed from episode_id hash if not provided
        if seed is None:
            seed = abs(hash(episode_id)) % (2 ** 31)

        # 3. Determine difficulty from task_id
        difficulty_map = {
            'task_easy':   'easy',
            'task_medium': 'medium',
            'task_hard':   'hard',
        }
        if task_id is None:
            task_id = 'task_easy'
        difficulty = difficulty_map.get(task_id, 'easy')

        # 4. Generate case
        case = self._case_gen.generate_case(difficulty, seed)

        # 5. Store session
        _SESSIONS[episode_id] = {
            'task_id':             task_id,
            'difficulty':          difficulty,
            'case':                case,
            'step':                0,
            'pre_jitter_accuracy': 0.0,
            'adv_stages_used':     0,
            'last_agent_traj':     None,
            'seed':                seed,
        }
        _LAST_EPISODE_ID = episode_id

        # 6. Build initial observation
        initial_config = case['initial_config']
        target_config  = case['target_config']
        baseline_traj  = case['baseline_trajectory']

        tooth_table      = self._build_tooth_table(initial_config, target_config)
        tooth_table_text = self._build_tooth_table_text(tooth_table)
        arch_graph_json  = self._build_arch_graph_json()
        baseline_json    = self._build_baseline_json(baseline_traj)

        task_desc = _TASK_DESCRIPTIONS[difficulty]

        return AlignerObservation(
            done=False,
            reward=None,
            task_id=task_id,
            current_stage=0,
            stages_remaining=N_STAGES,
            task_description=task_desc,
            tooth_table=tooth_table,
            tooth_table_text=tooth_table_text,
            arch_graph_json=arch_graph_json,
            baseline_trajectory_json=baseline_json,
            last_plan_feedback='',
            jitter_description='',
            step_number=0,
            adversarial_jitter_applied=False,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: AlignerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AlignerObservation:
        """
        Grade the agent's trajectory plan and return feedback.

        Step 1 (all difficulties): Grade submitted 24-stage plan.
        Step 2 (task_hard only):   Apply adversarial jitter and ask agent to revise.
        Episode done after step 1 (easy/medium) or step 2 (hard).
        """
        episode_id = _LAST_EPISODE_ID
        if episode_id is None or episode_id not in _SESSIONS:
            raise RuntimeError('No active session. Call /reset before /step.')

        session    = _SESSIONS[episode_id]
        case       = session['case']
        task_id    = session['task_id']
        difficulty = session['difficulty']
        step_num   = session['step'] + 1
        session['step'] = step_num

        initial_config = case['initial_config']   # (28, 7)
        target_config  = case['target_config']    # (28, 7)
        baseline_traj  = case['baseline_trajectory']  # (26, 28, 7)

        # --- Determine context depending on hard step 2 ---
        is_hard_step2 = (difficulty == 'hard' and step_num == 2)

        if is_hard_step2:
            # Step 2: agent revises from the jitter point onward
            current_traj = session.get('jittered_traj', baseline_traj)
            jitter_stage = session.get('jitter_stage', 12)
            stages_remaining = N_STAGES - jitter_stage
        else:
            # Step 1: full 24-stage plan
            current_traj = baseline_traj
            jitter_stage = None
            stages_remaining = N_STAGES

        # --- Parse agent's trajectory ---
        agent_traj = self._parse_agent_trajectory(
            action, initial_config, target_config, stages_remaining
        )

        # For hard step 2, splice revised stages back into the current trajectory
        if is_hard_step2 and jitter_stage is not None:
            full_traj = current_traj.copy()
            revised_count = agent_traj.shape[0] - 2  # excludes stage 0 and 25 padding
            for s in range(revised_count):
                global_stage = jitter_stage + 1 + s
                if global_stage < 25:
                    full_traj[global_stage] = agent_traj[1 + s]
            agent_traj_for_grade = full_traj
        else:
            agent_traj_for_grade = agent_traj

        # --- Grade ---
        reward, feedback = self._grader.grade(
            task_id=task_id,
            agent_traj=agent_traj_for_grade,
            initial=initial_config,
            target=target_config,
            adv_stages=session['adv_stages_used'],
            pre_jitter_accuracy=session['pre_jitter_accuracy'],
        )

        # Store agent trajectory in session
        session['last_agent_traj'] = agent_traj_for_grade

        # --- task_hard step 1: apply adversarial jitter ---
        adv_jitter_applied  = False
        jittered_stage_out  = None
        jittered_teeth_out  = None

        if difficulty == 'hard' and step_num == 1:
            # Record pre-jitter accuracy for step 2 grading
            session['pre_jitter_accuracy'] = reward

            # Apply jitter to a mid-point stage
            jitter_stage = 12
            rng = np.random.default_rng(session['seed'] + 1)
            jitter_strength = 0.2
            jittered_traj, jittered_teeth = self._case_gen.apply_adversarial_jitter(
                agent_traj_for_grade, jitter_stage, jitter_strength, rng
            )
            session['jittered_traj']   = jittered_traj
            session['jitter_stage']    = jitter_stage
            session['jittered_teeth']  = jittered_teeth
            session['adv_stages_used'] = 1   # jitter applied; enables recovery bonus
            adv_jitter_applied  = True
            jittered_stage_out  = jitter_stage
            jittered_teeth_out  = jittered_teeth

            # Build revised obs (done=False for hard, agent must submit step 2)
            current_config = jittered_traj[jitter_stage]
            tooth_table      = self._build_tooth_table(current_config, target_config)
            tooth_table_text = self._build_tooth_table_text(tooth_table)
            arch_graph_json  = self._build_arch_graph_json()
            baseline_json    = self._build_baseline_json(baseline_traj)

            return AlignerObservation(
                done=False,
                reward=None,   # no reward yet — episode continues
                task_id=task_id,
                current_stage=jitter_stage,
                stages_remaining=N_STAGES - jitter_stage,
                task_description=_TASK_DESCRIPTIONS[difficulty],
                tooth_table=tooth_table,
                tooth_table_text=tooth_table_text,
                arch_graph_json=arch_graph_json,
                baseline_trajectory_json=baseline_json,
                last_plan_feedback=feedback,
                jitter_description=(
                    f'Stage {jitter_stage} was perturbed on teeth: {jittered_teeth}. '
                    f'Revise stages {jitter_stage+1}-24.'
                ),
                step_number=step_num,
                adversarial_jitter_applied=True,
            )

        # --- Determine done ---
        done = True  # easy and medium always done after step 1
        if difficulty == 'hard' and step_num < 2:
            done = False
        if step_num >= self.MAX_STEPS:
            done = True

        # --- Build final observation ---
        tooth_table      = self._build_tooth_table(initial_config, target_config)
        tooth_table_text = self._build_tooth_table_text(tooth_table)
        arch_graph_json  = self._build_arch_graph_json()
        baseline_json    = self._build_baseline_json(baseline_traj)

        return AlignerObservation(
            done=done,
            reward=reward,
            task_id=task_id,
            current_stage=N_STAGES if not is_hard_step2 else (jitter_stage or N_STAGES),
            stages_remaining=0,
            task_description=_TASK_DESCRIPTIONS[difficulty],
            tooth_table=tooth_table,
            tooth_table_text=tooth_table_text,
            arch_graph_json=arch_graph_json,
            baseline_trajectory_json=baseline_json,
            last_plan_feedback=feedback,
            jitter_description='',
            step_number=step_num,
            adversarial_jitter_applied=adv_jitter_applied,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> AlignerState:
        """Current episode state."""
        if _LAST_EPISODE_ID and _LAST_EPISODE_ID in _SESSIONS:
            session = _SESSIONS[_LAST_EPISODE_ID]
            return AlignerState(
                episode_id=_LAST_EPISODE_ID,
                step_count=session['step'],
                task_id=session['task_id'],
                difficulty=session['difficulty'],
                seed=session['seed'],
                current_stage=min(session['step'] * N_STAGES, N_STAGES),
                total_violations=0,
                adversarial_perturbations=session['adv_stages_used'],
                best_trajectory_score=session['pre_jitter_accuracy'],
            )
        return AlignerState()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tooth_table(
        self,
        current_config: np.ndarray,
        target_config: np.ndarray,
    ) -> List[ToothPoseTableRow]:
        """Build list of 28 ToothPoseTableRow entries."""
        rows = []
        for i, tooth_id in enumerate(TOOTH_IDS):
            curr = current_config[i]   # (7,)
            tgt  = target_config[i]    # (7,)

            # Translation distance (mm)
            dist_mm = float(np.linalg.norm(curr[4:7] - tgt[4:7]))

            # Rotation distance (degrees): angle of relative rotation
            q_curr = curr[:4]
            q_tgt  = tgt[:4]
            q_inv  = quaternion_inverse(q_curr)
            q_rel  = quaternion_multiply(q_tgt, q_inv)
            dist_deg = quaternion_to_angle_deg(q_rel)

            rows.append(ToothPoseTableRow(
                tooth_id=tooth_id,
                tooth_type=TOOTH_TYPES[tooth_id],
                current_qw=float(curr[0]), current_qx=float(curr[1]),
                current_qy=float(curr[2]), current_qz=float(curr[3]),
                current_tx=float(curr[4]), current_ty=float(curr[5]),
                current_tz=float(curr[6]),
                target_qw=float(tgt[0]), target_qx=float(tgt[1]),
                target_qy=float(tgt[2]), target_qz=float(tgt[3]),
                target_tx=float(tgt[4]), target_ty=float(tgt[5]),
                target_tz=float(tgt[6]),
                remaining_trans_mm=round(dist_mm, 3),
                remaining_rot_deg=round(dist_deg, 3),
            ))
        return rows

    def _build_tooth_table_text(
        self, tooth_table: List[ToothPoseTableRow]
    ) -> str:
        """Build a markdown table for human-readable display."""
        header = (
            '| Tooth | Type              | CurrPos(mm)            '
            '| TargetPos(mm)          | Dist_mm | Dist_deg |\n'
            '|-------|-------------------|------------------------|'
            '------------------------|---------|----------|\n'
        )
        lines = [header]
        for row in tooth_table:
            curr_str   = f'({row.current_tx:.1f}, {row.current_ty:.1f}, {row.current_tz:.1f})'
            target_str = f'({row.target_tx:.1f}, {row.target_ty:.1f}, {row.target_tz:.1f})'
            lines.append(
                f'| {row.tooth_id:5d} | {row.tooth_type:17s} | {curr_str:22s} '
                f'| {target_str:22s} | {row.remaining_trans_mm:7.3f} | {row.remaining_rot_deg:8.3f} |\n'
            )
        return ''.join(lines)

    def _build_arch_graph_json(self) -> str:
        """Build adjacency list from ARCH_ADJACENCY, serialised to JSON."""
        adjacency: Dict[int, List[int]] = {}
        for (a, b) in ARCH_ADJACENCY:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        # JSON keys must be strings
        return json.dumps({str(k): v for k, v in sorted(adjacency.items())})

    def _build_baseline_json(self, baseline_trajectory: np.ndarray) -> str:
        """Serialise stages 1-24 of the SLERP baseline to compact JSON."""
        # baseline_trajectory shape: (26, 28, 7)
        stages = {}
        for s in range(1, 25):
            stages[str(s)] = baseline_trajectory[s].tolist()
        return json.dumps(stages, separators=(',', ':'))

    def _parse_agent_trajectory(
        self,
        action: AlignerAction,
        initial_config: np.ndarray,
        target_config: np.ndarray,
        stages_remaining: int,
    ) -> np.ndarray:
        """
        Parse action.trajectory into a numpy array of shape (26, 28, 7).

        - action.trajectory is expected to be a dict keyed by stage number (int or str)
          mapping to a list of 28 poses each of length 7, OR a list of 24 (or fewer)
          stages where each stage is a list of 28 x 7 vectors.
        - Quaternions are normalised.
        - Padded: stage 0 = initial_config, stage 25 = target_config.
        """
        traj = np.zeros((26, N_TEETH, 7), dtype=np.float64)
        traj[0]  = initial_config.copy()
        traj[25] = target_config.copy()

        # action.trajectory is List[ToothTrajectoryStage]
        # Each stage has .stage_index (1-24) and .poses (list of 28 x 7-float lists)

        def _fill_stage(stage_idx: int, pose_list: Any) -> None:
            """Fill traj[stage_idx] from a list-of-28-pose-7-vectors."""
            if stage_idx < 1 or stage_idx > 24:
                return
            if not pose_list or len(pose_list) == 0:
                return
            arr = np.array(pose_list, dtype=np.float64)
            if arr.shape != (N_TEETH, 7):
                return
            # Normalise quaternions
            for i in range(N_TEETH):
                q = arr[i, :4]
                n = np.linalg.norm(q)
                if n > 1e-10:
                    arr[i, :4] = q / n
                else:
                    arr[i, :4] = initial_config[i, :4]
            traj[stage_idx] = arr

        raw = action.trajectory
        if raw:
            for stage_obj in raw:
                # stage_obj is a ToothTrajectoryStage (Pydantic model)
                if hasattr(stage_obj, 'stage_index') and hasattr(stage_obj, 'poses'):
                    _fill_stage(int(stage_obj.stage_index), stage_obj.poses)
                elif isinstance(stage_obj, dict):
                    _fill_stage(int(stage_obj.get('stage_index', 0)), stage_obj.get('poses', []))

        # Fill any zero (unpopulated) stages with SLERP interpolation
        for s in range(1, 25):
            if np.allclose(traj[s], 0.0):
                alpha = s / 25.0
                for i in range(N_TEETH):
                    from server.quaternion_utils import quaternion_slerp, quaternion_normalize
                    traj[s, i, :4] = quaternion_normalize(
                        quaternion_slerp(traj[0, i, :4], traj[25, i, :4], alpha)
                    )
                    traj[s, i, 4:7] = (1.0 - alpha) * traj[0, i, 4:7] + alpha * traj[25, i, 4:7]

        return traj


# ---------------------------------------------------------------------------
# Stepwise 24-step sequential environment
# ---------------------------------------------------------------------------

_STEPWISE_SESSIONS: Dict[str, dict] = {}

STEPWISE_TASK_DESCRIPTION = """DENTAL ALIGNER PLANNING — STEPWISE MODE (battisiBot v2)
You are an expert orthodontic treatment planning AI.
Plan ONE aligner stage at a time (24 sequential decisions).

Each step: observe current tooth poses → submit 28 poses for the next stage.
You receive per-step feedback (dense reward) after each commit.

Tools available:
  inspect_tooth(tooth_id)  — get detailed pose, distance to target, neighbors
  simulate_step(poses)     — preview violations & reward WITHOUT committing
  check_collisions()       — detect inter-tooth penetration pairs
  commit_stage(poses)      — finalize the stage (irreversible, advances episode)
  rollback_stage()         — undo last commit (max 2 per episode)

CONSTRAINTS: max 0.25mm translation, 2.0° rotation per tooth per stage.
SCORING: progress 40% + compliance 30% + smoothness 20% + staging 10% per step.
""".strip()


class StepwiseDentalEnvironment:
    """
    24-step sequential RL environment for dental aligner trajectory planning.

    Each episode = 24 steps. At each step the agent commits tooth poses for
    one aligner stage and receives a dense reward. Tool-use actions allow
    the agent to inspect, simulate, and rollback before committing.
    """

    MAX_ROLLBACKS = 2

    def __init__(self):
        self._case_gen = DentalCaseGenerator()
        self._grader = AlignerGrader()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = 'task_easy',
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        source: str = 'synthetic',
        patient_path: Optional[str] = None,
        difficulty_params: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Reset environment and return initial observation."""
        from models import StepwiseObservation

        episode_id = episode_id or str(uuid.uuid4())
        seed = seed if seed is not None else hash(episode_id) % (2**31)

        # Parse difficulty from task_id
        difficulty = task_id.replace('task_', '') if task_id.startswith('task_') else 'easy'

        # Generate case from dataset or synthetic
        if source != 'synthetic' and patient_path:
            case = self._case_gen.generate_case_from_dataset(
                source=source, patient_path=patient_path,
                difficulty=difficulty, seed=seed,
            )
        else:
            case = self._case_gen.generate_case(difficulty, seed)

        initial = case['initial_config']   # (28, 7)
        target = case['target_config']     # (28, 7)

        # Trajectory buffer: stage 0 = initial, stage 25 = target, 1-24 filled by agent
        trajectory = np.zeros((26, N_TEETH, 7), dtype=np.float64)
        trajectory[0] = initial.copy()
        trajectory[25] = target.copy()

        session = {
            'episode_id': episode_id,
            'task_id': task_id,
            'case': case,
            'initial': initial,
            'target': target,
            'trajectory': trajectory,
            'current_stage': 0,
            'done': False,
            'seed': seed,
            'difficulty': difficulty,
            'source': source,
            'rollback_count': 0,
            'tool_call_count': 0,
            'reward_history': [],
            'cumulative_violations': 0,
        }
        _STEPWISE_SESSIONS[episode_id] = session

        return self._build_observation(session)

    # ------------------------------------------------------------------
    # step — commit one stage
    # ------------------------------------------------------------------

    def step(self, episode_id: str, poses: list) -> dict:
        """
        Commit tooth poses for the next stage.

        Args:
            episode_id: Session ID from reset()
            poses: 28x7 list of tooth poses for the next stage

        Returns:
            Observation dict with step_reward and updated state.
        """
        session = _STEPWISE_SESSIONS.get(episode_id)
        if session is None:
            return {'error': 'No active session. Call reset_stepwise first.'}
        if session['done']:
            return {'error': 'Episode already done.'}

        stage = session['current_stage']
        next_stage = stage + 1

        # Validate and normalize poses
        arr = np.array(poses, dtype=np.float64)
        if arr.shape != (N_TEETH, 7):
            return {'error': f'Expected ({N_TEETH}, 7) poses, got {arr.shape}'}

        for i in range(N_TEETH):
            n = np.linalg.norm(arr[i, :4])
            if n > 1e-10:
                arr[i, :4] /= n
            else:
                arr[i, :4] = session['initial'][i, :4]

        # Write into trajectory buffer
        session['trajectory'][next_stage] = arr

        # Compute per-step dense reward
        step_reward_info = self._compute_step_reward(session, next_stage)
        session['reward_history'].append(step_reward_info['step_reward'])
        session['cumulative_violations'] += step_reward_info['violations_this_step']

        # Advance stage
        session['current_stage'] = next_stage

        # Check if episode is done
        terminal_reward = None
        if next_stage >= N_STAGES:
            session['done'] = True
            # Fill remaining stages if < 24 committed (shouldn't happen but safety)
            self._fill_remaining_slerp(session)
            # Compute terminal reward via full grader
            score, feedback = self._grader.grade(
                session['task_id'],
                session['trajectory'],
                session['initial'],
                session['target'],
                adv_stages=0,
                pre_jitter_accuracy=0.0,
            )
            terminal_reward = score

        obs = self._build_observation(session)
        obs['step_reward'] = step_reward_info['step_reward']
        obs['reward_breakdown'] = step_reward_info
        obs['terminal_reward'] = terminal_reward
        return obs

    # ------------------------------------------------------------------
    # handle_tool — tool-use actions
    # ------------------------------------------------------------------

    def handle_tool(self, episode_id: str, tool: str, args: dict) -> dict:
        """Handle a tool-use action."""
        session = _STEPWISE_SESSIONS.get(episode_id)
        if session is None:
            return {'tool': tool, 'success': False, 'error': 'No active session.'}
        if session['done']:
            return {'tool': tool, 'success': False, 'error': 'Episode done.'}

        session['tool_call_count'] += 1

        if tool == 'inspect_tooth':
            return self._tool_inspect_tooth(session, args)
        elif tool == 'simulate_step':
            return self._tool_simulate_step(session, args)
        elif tool == 'check_collisions':
            return self._tool_check_collisions(session, args)
        elif tool == 'commit_stage':
            # commit_stage is equivalent to step()
            result = self.step(episode_id, args.get('poses', []))
            return {'tool': tool, 'success': True, 'result': result}
        elif tool == 'rollback_stage':
            return self._tool_rollback(session)
        else:
            return {'tool': tool, 'success': False, 'error': f'Unknown tool: {tool}'}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_inspect_tooth(self, session: dict, args: dict) -> dict:
        """Return detailed info about a specific tooth."""
        tooth_id = args.get('tooth_id')
        if tooth_id not in TOOTH_IDS:
            return {'tool': 'inspect_tooth', 'success': False, 'error': f'Invalid tooth_id: {tooth_id}'}

        idx = TOOTH_IDS.index(tooth_id)
        stage = session['current_stage']
        current = session['trajectory'][stage][idx]
        target = session['target'][idx]

        trans_dist = float(np.linalg.norm(current[4:7] - target[4:7]))
        q_rel = quaternion_multiply(target[:4], quaternion_inverse(current[:4]))
        rot_dist = quaternion_to_angle_deg(q_rel)

        # Compute per-stage budget remaining
        initial_trans_dist = float(np.linalg.norm(session['initial'][idx, 4:7] - target[4:7]))
        progress = 1.0 - (trans_dist / max(initial_trans_dist, 1e-6))

        # Neighbors from adjacency
        neighbors = []
        for a, b in ARCH_ADJACENCY:
            if a == tooth_id:
                neighbors.append(b)
            elif b == tooth_id:
                neighbors.append(a)

        return {
            'tool': 'inspect_tooth',
            'success': True,
            'result': {
                'tooth_id': tooth_id,
                'tooth_type': TOOTH_TYPES[tooth_id],
                'current_pose': current.tolist(),
                'target_pose': target.tolist(),
                'remaining_trans_mm': round(trans_dist, 4),
                'remaining_rot_deg': round(rot_dist, 4),
                'progress': round(progress, 4),
                'staging_priority': STAGING_PRIORITY.index(TOOTH_TYPES[tooth_id]) if TOOTH_TYPES[tooth_id] in STAGING_PRIORITY else 99,
                'neighbors': neighbors,
                'current_stage': stage,
            }
        }

    def _tool_simulate_step(self, session: dict, args: dict) -> dict:
        """Preview a step without committing — returns reward preview."""
        poses = args.get('poses', [])
        arr = np.array(poses, dtype=np.float64)
        if arr.shape != (N_TEETH, 7):
            return {'tool': 'simulate_step', 'success': False, 'error': f'Expected ({N_TEETH},7) poses'}

        # Normalize quaternions
        for i in range(N_TEETH):
            n = np.linalg.norm(arr[i, :4])
            if n > 1e-10:
                arr[i, :4] /= n

        # Temporarily write to trajectory, compute reward, then undo
        next_stage = session['current_stage'] + 1
        saved = session['trajectory'][next_stage].copy()
        session['trajectory'][next_stage] = arr
        reward_info = self._compute_step_reward(session, next_stage)
        session['trajectory'][next_stage] = saved  # restore

        return {
            'tool': 'simulate_step',
            'success': True,
            'result': {
                'preview_reward': reward_info['step_reward'],
                'progress': reward_info['progress'],
                'compliance': reward_info['compliance'],
                'smoothness': reward_info['smoothness'],
                'staging': reward_info['staging'],
                'violations': reward_info['violations_this_step'],
            }
        }

    def _tool_check_collisions(self, session: dict, args: dict) -> dict:
        """Check for inter-tooth collisions at current stage."""
        stage = session['current_stage']
        current = session['trajectory'][stage]
        collision_pairs = []
        threshold_mm = 0.5  # teeth closer than 0.5mm are "colliding"

        for i in range(N_TEETH):
            for j in range(i + 1, N_TEETH):
                dist = float(np.linalg.norm(current[i, 4:7] - current[j, 4:7]))
                if dist < threshold_mm:
                    collision_pairs.append({
                        'tooth_a': TOOTH_IDS[i],
                        'tooth_b': TOOTH_IDS[j],
                        'distance_mm': round(dist, 4),
                    })

        return {
            'tool': 'check_collisions',
            'success': True,
            'result': {
                'collision_pairs': collision_pairs,
                'n_collisions': len(collision_pairs),
            }
        }

    def _tool_rollback(self, session: dict) -> dict:
        """Undo the last committed stage."""
        if session['rollback_count'] >= self.MAX_ROLLBACKS:
            return {'tool': 'rollback_stage', 'success': False,
                    'error': f'Max rollbacks ({self.MAX_ROLLBACKS}) reached.'}
        if session['current_stage'] <= 0:
            return {'tool': 'rollback_stage', 'success': False, 'error': 'Nothing to rollback.'}

        stage = session['current_stage']
        session['trajectory'][stage] = np.zeros((N_TEETH, 7), dtype=np.float64)
        session['current_stage'] -= 1
        session['rollback_count'] += 1
        if session['reward_history']:
            session['reward_history'].pop()

        return {
            'tool': 'rollback_stage',
            'success': True,
            'result': {
                'rolled_back_to_stage': session['current_stage'],
                'rollbacks_remaining': self.MAX_ROLLBACKS - session['rollback_count'],
            }
        }

    # ------------------------------------------------------------------
    # Per-step dense reward
    # ------------------------------------------------------------------

    def _compute_step_reward(self, session: dict, stage: int) -> dict:
        """
        Compute dense reward for a single stage transition.

        Components (weighted):
          progress (0.4): how much closer teeth moved to target
          compliance (0.3): per-stage movement within clinical limits
          smoothness (0.2): consistency with previous step
          staging (0.1): incisors-before-molars bonus
        """
        from server.dental_constants import (
            MAX_TRANSLATION_PER_STAGE_MM, MAX_ROTATION_PER_STAGE_DEG,
        )

        traj = session['trajectory']
        target = session['target']
        prev_config = traj[stage - 1]  # previous stage
        curr_config = traj[stage]      # current (just committed)

        # --- Progress (0.4) ---
        prev_dists = np.linalg.norm(prev_config[:, 4:7] - target[:, 4:7], axis=1)
        curr_dists = np.linalg.norm(curr_config[:, 4:7] - target[:, 4:7], axis=1)
        total_prev = prev_dists.sum()
        progress = float((prev_dists - curr_dists).sum() / max(total_prev, 1e-6))
        progress = max(0.0, min(1.0, progress))

        # --- Compliance (0.3) ---
        violations = 0
        for i in range(N_TEETH):
            dt = float(np.linalg.norm(curr_config[i, 4:7] - prev_config[i, 4:7]))
            q_rel = quaternion_multiply(curr_config[i, :4], quaternion_inverse(prev_config[i, :4]))
            dr = quaternion_to_angle_deg(q_rel)
            if dt > MAX_TRANSLATION_PER_STAGE_MM * 1.05:  # 5% tolerance
                violations += 1
            if dr > MAX_ROTATION_PER_STAGE_DEG * 1.05:
                violations += 1
        max_violations = N_TEETH * 2
        compliance = 1.0 - (violations / max_violations)

        # --- Smoothness (0.2) ---
        if stage >= 2:
            prev_prev = traj[stage - 2]
            prev_deltas = np.linalg.norm(prev_config[:, 4:7] - prev_prev[:, 4:7], axis=1)
            curr_deltas = np.linalg.norm(curr_config[:, 4:7] - prev_config[:, 4:7], axis=1)
            delta_variance = float(np.var(curr_deltas - prev_deltas))
            smoothness = max(0.0, 1.0 - min(1.0, delta_variance / 0.05))
        else:
            smoothness = 1.0  # first stage, no comparison

        # --- Staging signal (0.1) ---
        # Bonus if incisors have moved more than molars up to this stage
        incisor_indices = [i for i, tid in enumerate(TOOTH_IDS)
                          if TOOTH_TYPES[tid] in ('central_incisor', 'lateral_incisor', 'canine')]
        molar_indices = [i for i, tid in enumerate(TOOTH_IDS)
                         if TOOTH_TYPES[tid] in ('molar_1', 'molar_2')]

        initial = session['initial']
        incisor_progress = sum(
            np.linalg.norm(initial[i, 4:7] - target[i, 4:7]) -
            np.linalg.norm(curr_config[i, 4:7] - target[i, 4:7])
            for i in incisor_indices
        )
        molar_progress = sum(
            np.linalg.norm(initial[i, 4:7] - target[i, 4:7]) -
            np.linalg.norm(curr_config[i, 4:7] - target[i, 4:7])
            for i in molar_indices
        )
        # Early stages: incisors should lead
        if stage <= 12 and incisor_progress > molar_progress:
            staging = 1.0
        elif stage > 12:
            staging = 0.8  # later stages, less important
        else:
            staging = 0.3

        step_reward = (
            0.4 * progress +
            0.3 * compliance +
            0.2 * smoothness +
            0.1 * staging
        )

        return {
            'step_reward': round(step_reward, 4),
            'progress': round(progress, 4),
            'compliance': round(compliance, 4),
            'smoothness': round(smoothness, 4),
            'staging': round(staging, 4),
            'violations_this_step': violations,
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, session: dict) -> dict:
        """Build observation dict from session state."""
        stage = session['current_stage']
        current = session['trajectory'][stage]
        target = session['target']
        initial = session['initial']

        # Per-tooth progress (0.0 to 1.0)
        initial_dists = np.linalg.norm(initial[:, 4:7] - target[:, 4:7], axis=1)
        current_dists = np.linalg.norm(current[:, 4:7] - target[:, 4:7], axis=1)
        per_tooth_progress = []
        for i in range(N_TEETH):
            if initial_dists[i] > 1e-6:
                per_tooth_progress.append(round(1.0 - current_dists[i] / initial_dists[i], 4))
            else:
                per_tooth_progress.append(1.0)

        # Build tooth table text
        tooth_table = self._parent_build_tooth_table(current, target)
        tooth_table_text = self._parent_build_tooth_table_text(tooth_table)

        # Stage history summary
        history_parts = []
        for s, r in enumerate(session['reward_history'], 1):
            history_parts.append(f'Stage {s}: reward={r:.3f}')
        history_summary = '; '.join(history_parts) if history_parts else 'No stages committed yet.'

        return {
            'task_id': session['task_id'],
            'current_stage': stage,
            'stages_remaining': N_STAGES - stage,
            'done': session['done'],
            'current_config': current.tolist(),
            'target_config': target.tolist(),
            'per_tooth_progress': per_tooth_progress,
            'cumulative_violations': session['cumulative_violations'],
            'step_reward': None,
            'terminal_reward': None,
            'reward_breakdown': None,
            'stage_history_summary': history_summary,
            'tooth_table_text': tooth_table_text,
            'data_source': session.get('source', 'synthetic'),
            'task_description': STEPWISE_TASK_DESCRIPTION,
        }

    def _parent_build_tooth_table(self, current, target):
        """Reuse tooth table builder from parent environment."""
        rows = []
        for i, tooth_id in enumerate(TOOTH_IDS):
            curr = current[i]
            tgt = target[i]
            dist_mm = float(np.linalg.norm(curr[4:7] - tgt[4:7]))
            q_rel = quaternion_multiply(tgt[:4], quaternion_inverse(curr[:4]))
            dist_deg = quaternion_to_angle_deg(q_rel)
            rows.append(ToothPoseTableRow(
                tooth_id=tooth_id, tooth_type=TOOTH_TYPES[tooth_id],
                current_qw=float(curr[0]), current_qx=float(curr[1]),
                current_qy=float(curr[2]), current_qz=float(curr[3]),
                current_tx=float(curr[4]), current_ty=float(curr[5]),
                current_tz=float(curr[6]),
                target_qw=float(tgt[0]), target_qx=float(tgt[1]),
                target_qy=float(tgt[2]), target_qz=float(tgt[3]),
                target_tx=float(tgt[4]), target_ty=float(tgt[5]),
                target_tz=float(tgt[6]),
                remaining_trans_mm=round(dist_mm, 3),
                remaining_rot_deg=round(dist_deg, 3),
            ))
        return rows

    def _parent_build_tooth_table_text(self, tooth_table):
        """Build markdown table from tooth table rows."""
        header = (
            '| Tooth | Type              | CurrPos(mm)            '
            '| TargetPos(mm)          | Dist_mm | Dist_deg |\n'
            '|-------|-------------------|------------------------|'
            '------------------------|---------|----------|\n'
        )
        lines = [header]
        for row in tooth_table:
            curr_str = f'({row.current_tx:.1f}, {row.current_ty:.1f}, {row.current_tz:.1f})'
            target_str = f'({row.target_tx:.1f}, {row.target_ty:.1f}, {row.target_tz:.1f})'
            lines.append(
                f'| {row.tooth_id:5d} | {row.tooth_type:17s} | {curr_str:22s} '
                f'| {target_str:22s} | {row.remaining_trans_mm:7.3f} | {row.remaining_rot_deg:8.3f} |\n'
            )
        return ''.join(lines)

    def _fill_remaining_slerp(self, session: dict) -> None:
        """Fill any uncommitted stages with SLERP interpolation."""
        from server.quaternion_utils import quaternion_slerp, quaternion_normalize
        traj = session['trajectory']
        for s in range(1, 25):
            if np.allclose(traj[s], 0.0):
                alpha = s / 25.0
                for i in range(N_TEETH):
                    traj[s, i, :4] = quaternion_normalize(
                        quaternion_slerp(traj[0, i, :4], traj[25, i, :4], alpha)
                    )
                    traj[s, i, 4:7] = (1.0 - alpha) * traj[0, i, 4:7] + alpha * traj[25, i, 4:7]
