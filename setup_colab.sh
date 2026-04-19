#!/bin/bash
# One-shot Colab setup script.
# Run after SSH-ing into Colab from VS Code:
#   bash setup_colab.sh

set -e

echo "=== 1. Install dependencies ==="
pip install -q unsloth trl==0.16.1 wandb
pip install -q fastapi uvicorn pydantic scipy numpy Pillow matplotlib requests trimesh
pip install -q openenv-core 2>/dev/null || echo "openenv-core not on PyPI (expected)"

echo ""
echo "=== 2. Clone repo ==="
if [ ! -d "dental-aligner-claw4s" ]; then
    git clone https://github.com/mehular0ra/dental-aligner-claw4s.git
fi
cd dental-aligner-claw4s

echo ""
echo "=== 3. Start server ==="
python start_server_colab.py &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 8

echo ""
echo "=== 4. Health check ==="
for i in $(seq 1 5); do
    if curl -s http://localhost:7860/health | grep -q healthy; then
        echo "Server is healthy!"
        break
    fi
    echo "  Attempt $i/5..."
    sleep 3
done

echo ""
echo "=== 5. Quick benchmark ==="
python benchmarks.py --quick

echo ""
echo "=== 6. Test GRPO ==="
python train_grpo.py --test --episodes 2

echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "To train:"
echo "  python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 20"
echo ""
echo "Or use Unsloth (recommended):"
echo "  python -c \"
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit', max_seq_length=1024, load_in_4bit=True)
print('Model loaded!')
\""
