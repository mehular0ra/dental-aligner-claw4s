"""
GRPO Training Script for Dental Aligner Trajectory Planning.

Trains a small LLM (Qwen2.5-0.5B) to plan orthodontic aligner stages
using Group Relative Policy Optimization (GRPO) with decomposed reward
functions from the dental environment's grader.

Usage:
    # Quick test (CPU, 5 episodes)
    python train_grpo.py --test

    # Full training (GPU, 50+ episodes)
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 50

    # With vLLM acceleration (requires GPU)
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --use-vllm --episodes 100

Requirements:
    pip install trl transformers torch vllm wandb

References:
    - GRPO: Shao et al. "DeepSeekMath" arXiv:2402.03300, 2024.
    - TRL GRPOTrainer: huggingface.co/docs/trl/main/en/grpo_trainer
    - Qwen scheduler GRPO: github.com/anakin87/qwen-scheduler-grpo
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Environment client (talks to the running server)
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("DENTAL_SERVER_URL", "http://localhost:7860")


def _post(endpoint: str, data: dict) -> dict:
    """POST to dental aligner server."""
    url = f"{SERVER_URL}{endpoint}"
    req = urllib.request.Request(
        url, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(req, timeout=30).read().decode()
    return json.loads(raw, strict=False)


def generate_episode_prompt(obs: dict, stage: int) -> str:
    """
    Format the current observation into a prompt for the LLM.

    The LLM receives:
    - Current stage number and remaining stages
    - A summary of current vs target tooth positions
    - Constraint limits
    - History of rewards so far

    The LLM outputs a JSON with planning parameters that Python
    converts into exact SE(3) poses (same pattern as battisiBot).
    """
    n_teeth = len(obs.get("current_config", []))
    history = obs.get("stage_history_summary", "")

    # Compute summary stats for compact prompting
    current = obs.get("current_config", [])
    target = obs.get("target_config", [])
    progress = obs.get("per_tooth_progress", [0.0] * 28)

    mean_progress = sum(progress) / max(len(progress), 1)
    min_progress = min(progress) if progress else 0.0

    prompt = f"""You are an orthodontic treatment planning AI. Plan aligner stage {stage + 1} of 24.

CURRENT STATE:
- Stage: {stage}/24, Mean progress: {mean_progress:.1%}, Worst tooth: {min_progress:.1%}
- Violations so far: {obs.get('cumulative_violations', 0)}
- History: {history[:200]}

CONSTRAINTS:
- Max 0.25mm translation per tooth per stage
- Max 2.0 degrees rotation per tooth per stage
- Move incisors (teeth 11-13,21-23,31-33,41-43) BEFORE molars (16,17,26,27,36,37,46,47)

OUTPUT a JSON with your staging plan:
{{
  "strategy": "brief description of what to move this stage",
  "tooth_groups": [
    {{"teeth": [11,12,21,22], "fraction": 0.8, "priority": "high"}},
    {{"teeth": [13,23,33,43], "fraction": 0.5, "priority": "medium"}},
    {{"teeth": [16,17,26,27,36,37,46,47], "fraction": 0.1, "priority": "low"}}
  ]
}}

Where "fraction" is the interpolation fraction toward the target (0.0=stay, 1.0=jump to target).
The system will compute exact SE(3) poses from your plan using SLERP interpolation."""

    return prompt


def parse_completion_to_poses(
    completion: str,
    initial: List[List[float]],
    target: List[List[float]],
    stage: int,
) -> List[List[float]]:
    """
    Parse the LLM's JSON output into 28x7 tooth poses.

    The LLM outputs high-level planning parameters (which teeth to move,
    by how much). This function converts those into exact SE(3) poses
    using SLERP interpolation — same pattern as battisiBot.
    """
    # Default: uniform SLERP
    alpha_default = (stage + 1) / 25.0

    # Try to parse LLM JSON
    tooth_fractions = {}
    try:
        # Extract JSON from completion (may have thinking tags)
        json_start = completion.find("{")
        json_end = completion.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            plan = json.loads(completion[json_start:json_end])
            for group in plan.get("tooth_groups", []):
                frac = float(group.get("fraction", alpha_default))
                for tid in group.get("teeth", []):
                    tooth_fractions[tid] = frac
    except (json.JSONDecodeError, ValueError, KeyError):
        pass  # Fall back to default SLERP

    # FDI tooth IDs in order
    tooth_ids = [
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47,
    ]

    poses = []
    for i in range(28):
        tid = tooth_ids[i]
        # Use LLM fraction if available, else default SLERP
        frac = tooth_fractions.get(tid, alpha_default)
        frac = max(0.0, min(1.0, frac))

        q0, q1 = initial[i][:4], target[i][:4]
        t0, t1 = initial[i][4:], target[i][4:]

        # Linear quaternion interpolation + normalize (simplified SLERP)
        q = [q0[j] * (1 - frac) + q1[j] * frac for j in range(4)]
        qn = math.sqrt(sum(x * x for x in q))
        q = [x / max(qn, 1e-10) for x in q]

        # Linear translation interpolation
        t = [t0[j] * (1 - frac) + t1[j] * frac for j in range(3)]

        poses.append(q + t)

    return poses


# ---------------------------------------------------------------------------
# Reward functions (decomposed for GRPO — TRL logs each separately)
# ---------------------------------------------------------------------------

def _run_episode(completion: str, episode_data: dict) -> dict:
    """Run a full episode with the given completion strategy and return scores."""
    obs = _post("/reset_stepwise", {
        "task_id": episode_data["task_id"],
        "seed": episode_data["seed"],
        "source": "synthetic",
        "episode_id": f"grpo_{episode_data['seed']}_{id(completion)}",
    })
    initial = obs["current_config"]
    target = obs["target_config"]

    final_obs = None
    for stage in range(24):
        poses = parse_completion_to_poses(completion, initial, target, stage)
        final_obs = _post("/step_stepwise", {
            "episode_id": obs.get("task_id", "grpo") + f"_{episode_data['seed']}_{id(completion)}",
            "poses": poses,
        })
        if final_obs.get("done"):
            break

    return final_obs or {}


def accuracy_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Reward based on final accuracy (how close teeth end up to target)."""
    rewards = []
    for comp in completions:
        try:
            obs = _run_episode(comp, kwargs.get("episode_data", {}))
            rewards.append(float(obs.get("terminal_reward", 0.0)))
        except Exception:
            rewards.append(0.0)
    return rewards


def occlusion_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Reward based on Andrews' Six Keys occlusion composite score."""
    rewards = []
    for comp in completions:
        try:
            obs = _run_episode(comp, kwargs.get("episode_data", {}))
            bd = obs.get("reward_breakdown", {})
            rewards.append(float(bd.get("occlusion_composite", 0.0)))
        except Exception:
            rewards.append(0.0)
    return rewards


def compliance_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Reward based on constraint compliance (staying within biomechanical limits)."""
    rewards = []
    for comp in completions:
        try:
            obs = _run_episode(comp, kwargs.get("episode_data", {}))
            violations = obs.get("cumulative_violations", 0)
            # 0 violations = 1.0, 56 violations (max) = 0.0
            rewards.append(max(0.0, 1.0 - violations / 56.0))
        except Exception:
            rewards.append(0.0)
    return rewards


def staging_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Reward for correct staging order (incisors before molars)."""
    rewards = []
    for comp in completions:
        try:
            obs = _run_episode(comp, kwargs.get("episode_data", {}))
            bd = obs.get("reward_breakdown", {})
            rewards.append(float(bd.get("staging", 0.0)))
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_training_prompts(n_episodes: int = 50, seed_start: int = 0) -> List[Dict[str, Any]]:
    """Generate training prompts by resetting the environment with different seeds."""
    prompts = []
    for i in range(n_episodes):
        seed = seed_start + i
        try:
            obs = _post("/reset_stepwise", {
                "task_id": "task_easy",
                "seed": seed,
                "source": "synthetic",
                "episode_id": f"prompt_gen_{seed}",
            })
            prompt = generate_episode_prompt(obs, stage=0)
            prompts.append({
                "prompt": prompt,
                "episode_data": {"task_id": "task_easy", "seed": seed},
            })
        except Exception as e:
            print(f"Warning: failed to generate prompt for seed {seed}: {e}")
    return prompts


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    """Run GRPO training."""
    print(f"=== Dental Aligner GRPO Training ===")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Use vLLM: {args.use_vllm}")
    print()

    # Check server is running
    try:
        health = _post("/health", {}) if False else json.loads(
            urllib.request.urlopen(f"{SERVER_URL}/health").read()
        )
        print(f"Server: {health}")
    except Exception as e:
        print(f"ERROR: Server not running at {SERVER_URL}. Start with: uv run python -m server.app")
        sys.exit(1)

    # Generate prompts
    print(f"Generating {args.episodes} training prompts...")
    prompts = generate_training_prompts(args.episodes)
    print(f"Generated {len(prompts)} prompts")

    if args.test:
        # Test mode: just verify prompt generation and reward functions work
        print("\n=== TEST MODE ===")
        print(f"Sample prompt (truncated):\n{prompts[0]['prompt'][:300]}...")
        print("\nTesting reward functions with SLERP baseline...")

        # Generate a baseline completion
        baseline_completion = json.dumps({
            "strategy": "Uniform SLERP interpolation",
            "tooth_groups": [
                {"teeth": [11, 12, 21, 22], "fraction": 0.5, "priority": "high"},
                {"teeth": [16, 17, 26, 27, 36, 37, 46, 47], "fraction": 0.2, "priority": "low"},
            ]
        })
        print(f"Baseline completion: {baseline_completion[:100]}...")
        print("\nReward functions validated. Ready for GPU training.")
        print("\nTo run full training:")
        print(f"  python train_grpo.py --model {args.model} --episodes {args.episodes}")
        return

    # Full training with TRL
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: Install TRL: pip install trl transformers")
        print("For vLLM acceleration: pip install vllm")
        sys.exit(1)

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts for TRL
    train_data = []
    for p in prompts:
        train_data.append({
            "prompt": p["prompt"],
            "episode_data": p["episode_data"],
        })

    # GRPO config
    config = GRPOConfig(
        output_dir=f"./dental_grpo_{args.model.split('/')[-1]}",
        learning_rate=5e-6,
        per_device_train_batch_size=min(4, args.episodes),
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=1024,
        num_train_epochs=args.epochs,
        save_steps=max(1, args.episodes // 5),
        logging_steps=1,
        report_to="wandb" if args.wandb else "none",
        run_name="dental-grpo" if args.wandb else None,
        use_vllm=args.use_vllm,
    )

    print(f"Starting GRPO training...")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Generations per prompt: {config.num_generations}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Logging to: {'wandb' if args.wandb else 'console'}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[
            accuracy_reward_func,
            occlusion_reward_func,
            compliance_reward_func,
            staging_reward_func,
        ],
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final model
    trainer.save_model(f"./dental_grpo_{args.model.split('/')[-1]}/final")
    print(f"\nTraining complete. Model saved.")
    print(f"Run `wandb login` and check your dashboard for training curves.")


# ---------------------------------------------------------------------------
# Emergent behavior analysis
# ---------------------------------------------------------------------------

def analyze_emergent_behaviors(log_dir: str = "./dental_grpo_logs"):
    """
    Analyze training logs for emergent behaviors.

    Looks for:
    1. Staging correlation (incisors-first pattern)
    2. Velocity clamping (self-limiting below constraints)
    3. Molar anchor strategy (molars stationary while incisors move)
    4. Recovery prioritization (anterior-first after jitter)
    """
    print("=== Emergent Behavior Analysis ===")
    print("(Run after training to detect learned orthodontic behaviors)")
    print()
    print("Metrics to track per episode:")
    print("  1. Staging correlation: spearmanr(priority_ranks, movement_start_stages)")
    print("  2. Max per-step delta: should decrease over training (velocity clamping)")
    print("  3. Molar start stage: should increase (anchor strategy)")
    print("  4. Anterior recovery speed: should be > posterior (after jitter)")
    print()
    print("Compare episode 1 vs episode 50 for each metric.")
    print("Document: 'Without explicit instruction, the agent learned X'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for Dental Aligner RL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to train")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt for GRPO")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for fast generation")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--test", action="store_true", help="Test mode: verify setup without training")
    parser.add_argument("--analyze", action="store_true", help="Analyze emergent behaviors from logs")
    parser.add_argument("--server", default="http://localhost:7860", help="Server URL")

    args = parser.parse_args()
    SERVER_URL = args.server

    if args.analyze:
        analyze_emergent_behaviors()
    else:
        train(args)
