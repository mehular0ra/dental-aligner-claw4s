"""
One-shot Colab training script. Run this single file in a Colab cell:

    !git clone https://github.com/mehular0ra/dental-aligner-claw4s.git
    %cd dental-aligner-claw4s
    !python run_colab.py

Does everything: installs deps, stubs openenv, starts server, trains GRPO.
"""
import subprocess
import sys
import os
import time

# ============================================================
# Step 1: Install dependencies
# ============================================================
print("=" * 60)
print("STEP 1: Installing dependencies")
print("=" * 60)

packages = [
    "unsloth",
    "trl==0.16.1",
    "wandb",
    "fastapi",
    "uvicorn",
    "pydantic",
    "scipy",
    "numpy",
    "Pillow",
    "matplotlib",
    "requests",
    "trimesh",
]
for pkg in packages:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps" if pkg == "trl==0.16.1" else "", pkg],
        capture_output=True,
    )
print("Dependencies installed.")

# ============================================================
# Step 2: Stub openenv-core
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Stubbing openenv-core")
print("=" * 60)

import types

try:
    import openenv
    print("openenv-core found.")
except ImportError:
    openenv = types.ModuleType("openenv")
    openenv.core = types.ModuleType("openenv.core")
    openenv.core.env_server = types.ModuleType("openenv.core.env_server")

    class _Base:
        pass

    imod = types.ModuleType("openenv.core.env_server.interfaces")
    imod.Environment = _Base
    tmod = types.ModuleType("openenv.core.env_server.types")
    tmod.Action = _Base
    tmod.Observation = _Base
    tmod.State = _Base
    openenv.core.env_server.interfaces = imod
    openenv.core.env_server.types = tmod

    def _create_app(env, **kw):
        from fastapi import FastAPI
        return FastAPI()

    openenv.core.env_server.create_fastapi_app = _create_app

    for k, v in {
        "openenv": openenv,
        "openenv.core": openenv.core,
        "openenv.core.env_server": openenv.core.env_server,
        "openenv.core.env_server.interfaces": imod,
        "openenv.core.env_server.types": tmod,
    }.items():
        sys.modules[k] = v
    print("openenv-core stubbed.")

# ============================================================
# Step 3: Start server (in-process, background thread)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Starting dental aligner server")
print("=" * 60)

import threading
import uvicorn

def _run_server():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="warning")

server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()
print("Server starting in background thread...")
time.sleep(5)

# Health check
import json
import math
import urllib.request

SERVER = "http://localhost:7860"

def post(endpoint, data):
    req = urllib.request.Request(
        f"{SERVER}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode(), strict=False)

for attempt in range(10):
    try:
        resp = json.loads(urllib.request.urlopen(f"{SERVER}/health", timeout=5).read())
        print(f"Server ready: {resp}")
        break
    except Exception as e:
        if attempt < 9:
            print(f"  Attempt {attempt + 1}/10: waiting...")
            time.sleep(3)
        else:
            print(f"FATAL: Server failed to start after 10 attempts.")
            sys.exit(1)

# ============================================================
# Step 4: Quick benchmark
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Running benchmark")
print("=" * 60)

from benchmarks import run_benchmarks
results = run_benchmarks(quick=True)

# ============================================================
# Step 5: Generate training prompts
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Generating training prompts")
print("=" * 60)

N_EPISODES = 20
TOOTH_IDS = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,
             31,32,33,34,35,36,37,41,42,43,44,45,46,47]

def make_prompt(obs, stage=0):
    progress = obs.get("per_tooth_progress", [0] * 28)
    mean_p = sum(progress) / max(len(progress), 1)
    min_p = min(progress) if progress else 0
    return f"""You are an orthodontic treatment planning AI. Plan aligner stage {stage+1}/24.

STATE: Stage {stage}/24, Progress: {mean_p:.0%}, Worst: {min_p:.0%}, Violations: {obs.get('cumulative_violations',0)}

RULES: Max 0.25mm/2deg per tooth per stage. Move incisors BEFORE molars.

Output JSON: {{"strategy": "...", "tooth_groups": [{{"teeth": [...], "fraction": 0.0-1.0}}]}}"""

difficulties = [
    {"n_perturbed_teeth": 6, "translation_magnitude": 2.0},
    {"n_perturbed_teeth": 12, "translation_magnitude": 3.5},
    {"n_perturbed_teeth": 18, "translation_magnitude": 5.0},
    {"n_perturbed_teeth": 10, "translation_magnitude": 3.0, "jitter_probability": 0.2},
]

prompts = []
for i in range(N_EPISODES):
    seed = i * 7 + 42
    diff = difficulties[i % len(difficulties)]
    obs = post("/reset_stepwise", {
        "task_id": "task_easy", "seed": seed, "source": "synthetic",
        "episode_id": f"prompt_{i}", "difficulty_params": diff,
    })
    prompts.append({"prompt": make_prompt(obs), "seed": seed, "difficulty_params": diff})

print(f"Generated {len(prompts)} training prompts")

# ============================================================
# Step 6: Define reward functions
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Setting up reward functions")
print("=" * 60)

def run_episode(completion_text, seed, difficulty_params):
    tooth_fractions = {}
    try:
        js = completion_text[completion_text.find("{"):completion_text.rfind("}") + 1]
        plan = json.loads(js)
        for g in plan.get("tooth_groups", []):
            for tid in g.get("teeth", []):
                tooth_fractions[tid] = float(g.get("fraction", 0.5))
    except:
        pass

    eid = f"grpo_{seed}_{hash(completion_text) % 99999}"
    obs = post("/reset_stepwise", {
        "task_id": "task_easy", "seed": seed, "source": "synthetic",
        "episode_id": eid, "difficulty_params": difficulty_params,
    })
    init, tgt = obs["current_config"], obs["target_config"]

    final = None
    for s in range(1, 25):
        a_def = s / 25.0
        poses = []
        for i in range(28):
            f = tooth_fractions.get(TOOTH_IDS[i], a_def)
            f = max(0.0, min(1.0, f))
            q = [init[i][j] * (1 - f) + tgt[i][j] * f for j in range(4)]
            qn = math.sqrt(sum(x * x for x in q))
            q = [x / max(qn, 1e-10) for x in q]
            t = [init[i][4 + j] * (1 - f) + tgt[i][4 + j] * f for j in range(3)]
            poses.append(q + t)
        final = post("/step_stepwise", {"episode_id": eid, "poses": poses})
    return final

def reward_terminal(completions, **kwargs):
    rewards = []
    for i, comp in enumerate(completions):
        try:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            idx = i % len(prompts)
            obs = run_episode(text, prompts[idx]["seed"], prompts[idx]["difficulty_params"])
            rewards.append(float(obs.get("terminal_reward", 0.0)))
        except:
            rewards.append(0.0)
    return rewards

def reward_occlusion(completions, **kwargs):
    rewards = []
    for i, comp in enumerate(completions):
        try:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            idx = i % len(prompts)
            obs = run_episode(text, prompts[idx]["seed"], prompts[idx]["difficulty_params"])
            rewards.append(float(obs.get("reward_breakdown", {}).get("occlusion_composite", 0.0)))
        except:
            rewards.append(0.0)
    return rewards

# Quick test
test = json.dumps({"strategy": "test", "tooth_groups": [{"teeth": [11, 21], "fraction": 0.5}]})
test_r = reward_terminal([test])
print(f"Test reward: {test_r}")

# ============================================================
# Step 7: wandb (optional)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: wandb setup")
print("=" * 60)

USE_WANDB = False
try:
    import wandb
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        wandb.login(key=key)
        USE_WANDB = True
        print("wandb logged in from env var.")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("No WANDB_API_KEY set. Training without wandb.")
except ImportError:
    os.environ["WANDB_DISABLED"] = "true"
    print("wandb not installed. Training without it.")

# ============================================================
# Step 8: Load model with Unsloth
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Loading model with Unsloth")
print("=" * 60)

import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
print(f"Model loaded: {MODEL_NAME}")

# ============================================================
# Step 9: GRPO Training
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: GRPO TRAINING")
print("=" * 60)

from trl import GRPOConfig, GRPOTrainer

train_dataset = [{"prompt": [{"role": "user", "content": p["prompt"]}]} for p in prompts]

config = GRPOConfig(
    output_dir="./dental_grpo_output",
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=5,
    logging_steps=1,
    report_to="wandb" if USE_WANDB else "none",
    run_name="dental-grpo-unsloth" if USE_WANDB else None,
    bf16=True,
    gradient_accumulation_steps=2,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_terminal, reward_occlusion],
    args=config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print(f"Training {len(train_dataset)} episodes, batch={config.per_device_train_batch_size}, gens={config.num_generations}")
print("Starting...")
trainer.train()

# ============================================================
# Step 10: Save & Evaluate
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Saving model")
print("=" * 60)

model.save_pretrained("./dental_grpo_output/lora")
tokenizer.save_pretrained("./dental_grpo_output/lora")
print("LoRA adapters saved to ./dental_grpo_output/lora")

# Evaluate
FastLanguageModel.for_inference(model)
test_obs = post("/reset_stepwise", {"task_id": "task_easy", "seed": 999, "source": "synthetic", "episode_id": "eval_final"})
test_prompt = make_prompt(test_obs)
inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"\nModel output:\n{completion[:400]}")
result = run_episode(completion, 999, {"n_perturbed_teeth": 10, "translation_magnitude": 3.0})
print(f"\nTrained model reward: {result.get('terminal_reward', 0):.4f}")
print(f"SLERP baseline: ~0.87")
print(f"Occlusion: {result.get('reward_breakdown', {}).get('occlusion_composite', 0):.4f}")

print("\n" + "=" * 60)
print("DONE! Training complete.")
if USE_WANDB:
    print(f"wandb: {wandb.run.get_url()}")
print("=" * 60)
