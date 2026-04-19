"""
Pydantic models for the Dental Aligner Trajectory Planning environment.
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server.types import Action, Observation, State


class ToothTrajectoryStage(BaseModel):
    """A single stage in the agent's planned trajectory."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    stage_index: int = Field(..., ge=1, le=24, description='Stage 1-24 (0=initial, 25=final are fixed)')
    poses: List[List[float]] = Field(..., description='28 poses, each [qw,qx,qy,qz,tx,ty,tz]')
    tooth_ids: List[int] = Field(..., description='28 FDI tooth IDs in same order as poses')


class AlignerAction(Action):
    """Agent action: the full planned trajectory or revised remaining stages."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True, arbitrary_types_allowed=True)

    trajectory: List[ToothTrajectoryStage] = Field(
        default_factory=list,
        description='Agent planned stages. Length 24 for full plan, or stages_remaining for revised plan.'
    )
    reasoning: str = Field(default='', description="Agent's planning rationale")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ToothPoseTableRow(BaseModel):
    """One row in the tooth pose table — current vs target for one tooth."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    tooth_id: int
    tooth_type: str
    current_qw: float
    current_qx: float
    current_qy: float
    current_qz: float
    current_tx: float
    current_ty: float
    current_tz: float
    target_qw: float
    target_qx: float
    target_qy: float
    target_qz: float
    target_tx: float
    target_ty: float
    target_tz: float
    remaining_trans_mm: float = Field(..., description='Euclidean distance to target in mm')
    remaining_rot_deg: float = Field(..., description='Angular distance to target in degrees')


class AlignerObservation(Observation):
    """Observation returned to agent after reset() or step()."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True, arbitrary_types_allowed=True)
    # done: bool and reward: float|None are INHERITED from Observation

    current_stage: int = Field(default=0, description='Current stage index (0=initial)')
    stages_remaining: int = Field(default=24, description='Number of stages the agent must still plan')
    task_id: str = Field(default='')
    task_description: str = Field(default='')
    tooth_table: List[ToothPoseTableRow] = Field(default_factory=list)
    tooth_table_text: str = Field(default='', description='Markdown table for text-based agents')
    arch_graph_json: str = Field(default='', description='JSON adjacency list for GNN agents')
    baseline_trajectory_json: str = Field(default='', description='SLERP baseline as JSON reference')
    adversarial_jitter_applied: bool = Field(default=False)
    jitter_description: str = Field(default='')
    last_plan_feedback: str = Field(default='')
    step_number: int = Field(default=0)


class AlignerState(State):
    """Environment state (episode-level tracking)."""
    # episode_id and step_count are INHERITED; State has extra='allow'

    task_id: str = Field(default='')
    difficulty: str = Field(default='easy')
    current_stage: int = Field(default=0)
    seed: int = Field(default=0)
    total_violations: int = Field(default=0)
    adversarial_perturbations: int = Field(default=0)
    best_trajectory_score: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Stepwise (24-step sequential) models
# ---------------------------------------------------------------------------

class StepwiseAction(BaseModel):
    """Agent action for one stage in stepwise mode: 28 tooth poses."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    poses: List[List[float]] = Field(
        ..., description='28 poses for the next stage, each [qw,qx,qy,qz,tx,ty,tz]'
    )


class StepwiseObservation(BaseModel):
    """Observation returned after each step in stepwise mode."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    task_id: str = Field(default='')
    current_stage: int = Field(default=0, description='Current stage (0-24)')
    stages_remaining: int = Field(default=24)
    done: bool = Field(default=False)

    # Pose data
    current_config: List[List[float]] = Field(
        default_factory=list, description='Current 28x7 tooth poses after last committed stage'
    )
    target_config: List[List[float]] = Field(
        default_factory=list, description='Target 28x7 tooth poses (goal)'
    )

    # Progress tracking
    per_tooth_progress: List[float] = Field(
        default_factory=list, description='Per-tooth % of distance covered toward target (0.0-1.0)'
    )
    cumulative_violations: int = Field(default=0, description='Total constraint violations so far')

    # Rewards
    step_reward: Optional[float] = Field(default=None, description='Dense reward for last step')
    terminal_reward: Optional[float] = Field(default=None, description='Final graded score (only when done)')
    reward_breakdown: Optional[dict] = Field(default=None, description='Per-component reward details')

    # Context
    stage_history_summary: str = Field(
        default='', description='Text summary of movement so far'
    )
    tooth_table_text: str = Field(default='', description='Markdown table of current vs target poses')
    data_source: str = Field(default='synthetic', description='Dataset source used for this episode')


class ToolCall(BaseModel):
    """Agent tool call in stepwise mode."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    tool: str = Field(
        ..., description='Tool name: inspect_tooth, simulate_step, check_collisions, commit_stage, rollback_stage'
    )
    args: dict = Field(default_factory=dict, description='Tool arguments')


class ToolResult(BaseModel):
    """Result of a tool call."""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    tool: str
    success: bool = True
    result: dict = Field(default_factory=dict)
    error: Optional[str] = None
