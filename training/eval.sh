#!/bin/bash
# Wait for 4k-v2 training to finish, then auto merge + evaluate.
set -e

export FLASHINFER_DISABLE_VERSION_CHECK=1

# ===== Configure paths before running =====
TRAJTOM_ROOT=/path/to/TrajToM
CONDA_ENV_ROOT=/path/to/conda/envs
BASE_MODEL=/path/to/base-model   # e.g. Qwen/Qwen3.5-9B
# ==========================================

ADAPTER=$TRAJTOM_ROOT/training/saves/trajtom-grpo/adapter_model.safetensors

echo "Waiting for training to finish..."
echo "Checking: $ADAPTER"
echo ""

while [ ! -f "$ADAPTER" ]; do
  echo "  $(date '+%H:%M') - still training..."
  sleep 300
done

echo ""
echo "Training complete! Starting merge + eval pipeline..."
echo ""

# Step 1: Language-only merge
echo "=== Step 1: Language-only merge ==="
conda run -p $CONDA_ENV_ROOT/GRPO --no-capture-output python - <<EOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = '$BASE_MODEL'
SFT_ADAPTER = '$TRAJTOM_ROOT/training/LlamaFactory/saves/trajtom-sft-lora'
GRPO_ADAPTER = '$TRAJTOM_ROOT/training/saves/trajtom-grpo'
OUTPUT_DIR = '$TRAJTOM_ROOT/training/saves/trajtom-grpo-merged'

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
print('Applying SFT adapter...')
model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = model.merge_and_unload()
print('Applying GRPO 4k-v2 adapter...')
model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
model = model.merge_and_unload()
print(f'Saving to {OUTPUT_DIR}...')
model.save_pretrained(OUTPUT_DIR)
AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True).save_pretrained(OUTPUT_DIR)
print('Done.')
EOF

# Step 2: Weight surgery (add vision shell)
echo ""
echo "=== Step 2: Weight surgery ==="
conda run -p $CONDA_ENV_ROOT/llamafactory --no-capture-output python - <<EOF
import os, shutil, glob
from safetensors.torch import load_file, save_file
from safetensors import safe_open

SFT_MERGED = '$TRAJTOM_ROOT/training/LlamaFactory/saves/trajtom-sft-merged'
GRPO_MERGED = '$TRAJTOM_ROOT/training/saves/trajtom-grpo-merged'
OUTPUT_DIR = '$TRAJTOM_ROOT/training/saves/trajtom-grpo-mm'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Extracting visual weights...')
sft_vision = {}
for shard in sorted(glob.glob(f'{SFT_MERGED}/*.safetensors')):
    with safe_open(shard, framework='pt') as f:
        for key in f.keys():
            if 'visual' in key:
                sft_vision[key] = f.get_tensor(key)
print(f'  {len(sft_vision)} visual tensors')

print('Loading GRPO language weights...')
grpo_weights = {}
for fp in sorted(glob.glob(f'{GRPO_MERGED}/*.safetensors')):
    grpo_weights.update(load_file(fp))

combined = {**grpo_weights, **sft_vision}
save_file(combined, f'{OUTPUT_DIR}/model.safetensors', metadata={'format': 'pt'})

for f in ['config.json', 'generation_config.json', 'tokenizer.json',
          'tokenizer_config.json', 'chat_template.jinja', 'processor_config.json']:
    src = f'{SFT_MERGED}/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'{OUTPUT_DIR}/{f}')

idx = f'{OUTPUT_DIR}/model.safetensors.index.json'
if os.path.exists(idx):
    os.remove(idx)
print('Done.')
EOF

# Step 3: Evaluate
echo ""
echo "=== Step 3: Evaluate on test set ==="
cd $TRAJTOM_ROOT
conda run -p $CONDA_ENV_ROOT/GRPO --no-capture-output env FLASHINFER_DISABLE_VERSION_CHECK=1 python $TRAJTOM_ROOT/evaluate/evaluate_tom.py \
  --model-type vllm \
  --model-path $TRAJTOM_ROOT/training/saves/trajtom-grpo-mm \
  --data-path $TRAJTOM_ROOT/data/tom_questions_test.json \
  --dialogue-path $TRAJTOM_ROOT/data/tom_dialogues_test.json \
  --batch-size 256 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --output-dir $TRAJTOM_ROOT/evaluate/results/grpo-4k-v2 \
  --device cuda

cat $TRAJTOM_ROOT/evaluate/results/grpo-4k-v2/*/report.txt 2>/dev/null | head -5

