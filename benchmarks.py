"""
Benchmark suite for the dental aligner RL environment.

Runs SLERP baseline across multiple configurations and reports:
- Terminal reward, occlusion composite, PDL feasibility, collision score
- Comparison across difficulty levels, data sources, malocclusion classes
- Statistical summary (mean, std, min, max)

Usage:
    python benchmarks.py                 # Full benchmark (all configs)
    python benchmarks.py --quick         # Quick test (3 configs)
    python benchmarks.py --output results.json  # Save results to JSON
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from typing import Dict, List, Any

SERVER_URL = "http://localhost:7860"


def post(endpoint: str, data: dict) -> dict:
    url = f"{SERVER_URL}{endpoint}"
    req = urllib.request.Request(
        url, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    return json.loads(raw, strict=False)


def slerp_episode(episode_id: str, task_id: str, seed: int,
                   source: str = "synthetic", patient_path: str = None,
                   difficulty_params: dict = None) -> Dict[str, Any]:
    """Run a full 24-step SLERP baseline episode and return results."""
    reset_data = {
        "task_id": task_id, "seed": seed, "source": source,
        "episode_id": episode_id,
    }
    if patient_path:
        reset_data["patient_path"] = patient_path
    if difficulty_params:
        reset_data["difficulty_params"] = difficulty_params

    obs = post("/reset_stepwise", reset_data)
    init = obs["current_config"]
    tgt = obs["target_config"]

    final = None
    for s in range(1, 25):
        a = s / 25.0
        poses = []
        for i in range(28):
            q = [init[i][j] * (1 - a) + tgt[i][j] * a for j in range(4)]
            qn = math.sqrt(sum(x * x for x in q))
            q = [x / max(qn, 1e-10) for x in q]
            t = [init[i][4 + j] * (1 - a) + tgt[i][4 + j] * a for j in range(3)]
            poses.append(q + t)
        final = post("/step_stepwise", {"episode_id": episode_id, "poses": poses})

    bd = final.get("reward_breakdown", {})
    return {
        "terminal_reward": final.get("terminal_reward", 0.0),
        "occlusion_composite": bd.get("occlusion_composite", 0.0),
        "pdl_feasibility": bd.get("pdl_feasibility", 0.0),
        "collision_free": bd.get("collision_free", 0.0),
        "cumulative_violations": final.get("cumulative_violations", 0),
        "noncompliance_events": len([e for e in bd.get("noncompliance_event", []) or [] if e]),
    }


def run_benchmarks(quick: bool = False) -> List[Dict[str, Any]]:
    """Run all benchmark configurations."""
    results = []

    # --- 1. Difficulty levels ---
    print("=== Benchmark 1: Difficulty Levels (Synthetic) ===")
    for difficulty in ["easy", "medium", "hard"]:
        scores = []
        n_seeds = 3 if quick else 10
        for seed in range(n_seeds):
            eid = f"bench_diff_{difficulty}_{seed}"
            r = slerp_episode(eid, f"task_{difficulty}", seed)
            scores.append(r)
        avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}
        result = {"benchmark": "difficulty", "config": difficulty, "n_runs": n_seeds, **avg}
        results.append(result)
        print(f"  {difficulty:8s}: reward={avg['terminal_reward']:.4f}  occ={avg['occlusion_composite']:.3f}  pdl={avg['pdl_feasibility']:.2f}  coll={avg['collision_free']:.3f}")

    # --- 2. Adaptive difficulty ---
    print("\n=== Benchmark 2: Adaptive Difficulty Params ===")
    adaptive_configs = [
        {"name": "minimal", "params": {"n_perturbed_teeth": 4, "translation_magnitude": 1.0, "rotation_magnitude": 5.0}},
        {"name": "moderate", "params": {"n_perturbed_teeth": 12, "translation_magnitude": 3.0, "rotation_magnitude": 15.0}},
        {"name": "severe", "params": {"n_perturbed_teeth": 22, "translation_magnitude": 6.0, "rotation_magnitude": 30.0}},
        {"name": "adversarial", "params": {"n_perturbed_teeth": 16, "translation_magnitude": 4.0, "jitter_probability": 0.3}},
    ]
    for cfg in adaptive_configs:
        scores = []
        n_seeds = 2 if quick else 5
        for seed in range(n_seeds):
            eid = f"bench_adapt_{cfg['name']}_{seed}"
            r = slerp_episode(eid, "task_easy", seed, difficulty_params=cfg["params"])
            scores.append(r)
        avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}
        result = {"benchmark": "adaptive", "config": cfg["name"], "n_runs": n_seeds, **avg}
        results.append(result)
        print(f"  {cfg['name']:12s}: reward={avg['terminal_reward']:.4f}  occ={avg['occlusion_composite']:.3f}  violations={avg['cumulative_violations']:.1f}")

    # --- 3. Data sources ---
    print("\n=== Benchmark 3: Data Sources ===")
    data_configs = [
        {"name": "synthetic", "source": "synthetic", "path": None},
        {"name": "open_full_jaw", "source": "open_full_jaw", "path": "/tmp/ofj_p1/Patient_1"},
    ]
    # Check if Mendeley data exists
    import os
    mendeley_path = "/Users/mehul_fibr/Documents/Others/RL_projects/round1/datasets/mendeley-jaw/Human Lower Jaw Dataset/Teeth/STL"
    if os.path.isdir(mendeley_path):
        data_configs.append({"name": "mendeley_jaw", "source": "mendeley_jaw", "path": mendeley_path})

    for cfg in data_configs:
        try:
            eid = f"bench_data_{cfg['name']}"
            r = slerp_episode(eid, "task_easy", 42, source=cfg["source"], patient_path=cfg["path"])
            result = {"benchmark": "data_source", "config": cfg["name"], "n_runs": 1, **r}
            results.append(result)
            print(f"  {cfg['name']:15s}: reward={r['terminal_reward']:.4f}  occ={r['occlusion_composite']:.3f}  pdl={r['pdl_feasibility']:.2f}")
        except Exception as e:
            print(f"  {cfg['name']:15s}: SKIPPED ({e})")

    # --- 4. Summary statistics ---
    print("\n=== Summary ===")
    all_rewards = [r["terminal_reward"] for r in results]
    all_occ = [r["occlusion_composite"] for r in results]
    print(f"  Terminal reward: mean={sum(all_rewards)/len(all_rewards):.4f}, min={min(all_rewards):.4f}, max={max(all_rewards):.4f}")
    print(f"  Occlusion:       mean={sum(all_occ)/len(all_occ):.4f}, min={min(all_occ):.4f}, max={max(all_occ):.4f}")
    print(f"  Total configs tested: {len(results)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dental Aligner Environment Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer seeds)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--server", default="http://localhost:7860", help="Server URL")
    args = parser.parse_args()
    SERVER_URL = args.server

    start = time.time()
    results = run_benchmarks(quick=args.quick)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
