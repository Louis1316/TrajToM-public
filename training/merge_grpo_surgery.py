"""Weight surgery to build multimodal GRPO model:
1. GRPO-merged (from simple merge) already has updated language_model.* weights
2. SFT-merged has visual.* weights (unchanged since SFT didn't touch vision)
3. Combine: GRPO language weights + SFT vision weights + SFT multimodal config
"""
import os
import json
import shutil
import glob
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

SFT_MERGED = '/path/to/sft-merged'
GRPO_MERGED = '/path/to/grpo-pilot-merged'
OUTPUT_DIR = '/path/to/output/trajtom-grpo-pilot-mm'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Step 1: Loading all SFT-merged shards to extract visual weights...')
sft_vision_weights = {}
for shard_path in sorted(glob.glob(f'{SFT_MERGED}/*.safetensors')):
    with safe_open(shard_path, framework='pt') as f:
        for key in f.keys():
            if 'visual' in key:
                sft_vision_weights[key] = f.get_tensor(key)
print(f'  Extracted {len(sft_vision_weights)} visual tensors')

print('Step 2: Loading GRPO-merged (language-only) weights...')
grpo_files = sorted(glob.glob(f'{GRPO_MERGED}/*.safetensors'))
grpo_weights = {}
for fp in grpo_files:
    grpo_weights.update(load_file(fp))
print(f'  Loaded {len(grpo_weights)} language tensors')

print('Step 3: Combining weights...')
combined = {**grpo_weights, **sft_vision_weights}
print(f'  Total: {len(combined)} tensors')

print('Step 4: Saving combined safetensors (single shard for simplicity)...')
# Save as single file to avoid shard index complications
save_file(combined, f'{OUTPUT_DIR}/model.safetensors', metadata={'format': 'pt'})

print('Step 5: Copying multimodal config and tokenizer from SFT-merged...')
for f in ['config.json', 'generation_config.json', 'tokenizer.json',
          'tokenizer_config.json', 'chat_template.jinja', 'processor_config.json']:
    src = f'{SFT_MERGED}/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'{OUTPUT_DIR}/{f}')
        print(f'  copied {f}')

# Remove any stale index file (we're using single shard now)
idx = f'{OUTPUT_DIR}/model.safetensors.index.json'
if os.path.exists(idx):
    os.remove(idx)

print(f'Done. Saved to {OUTPUT_DIR}')
