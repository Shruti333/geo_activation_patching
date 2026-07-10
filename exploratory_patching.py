# %% [markdown]
# # Step-by-Step Exploratory Activation Patching Run

# %%
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer

# Import modules from our src directory
from src.patching import compute_logit_diff, patch_residual_stream_layer
from src.visualize import plot_patching_heatmap

# 1. Initialize environment and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Parse batch prompts from data directory
with open("../data/prompts.json", "r") as f:
    dataset = json.load(f)

clean_prompts = [item["clean_prompt"] for item in dataset]
corrupted_prompts = [item["corrupted_prompt"] for item in dataset]

clean_tokens = model.to_tokens(clean_prompts)
corrupted_tokens = model.to_tokens(corrupted_prompts)

correct_ids = torch.tensor([model.to_single_token(item["correct_token"]) for item in dataset], device=device)
incorrect_ids = torch.tensor([model.to_single_token(item["incorrect_token"]) for item in dataset], device=device)

# 3. Establish baselines
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

# Define our static metric tracking function
metric = lambda logits: compute_logit_diff(logits, correct_ids, incorrect_ids)

clean_baseline = metric(clean_logits).item()
corrupted_baseline = metric(corrupted_logits).item()

print(f"Clean Batch Baseline Logit Diff: {clean_baseline:.4f}")
print(f"Corrupted Batch Baseline Logit Diff: {corrupted_baseline:.4f}")

# 4. Execute the multidimensional patching sweep
n_layers = model.cfg.n_layers
n_tokens = clean_tokens.shape[1]
patching_matrix = np.zeros((n_layers, n_tokens))

for layer in range(n_layers):
    for pos in range(n_tokens):
        patched_val = patch_residual_stream_layer(
            model, clean_cache, corrupted_tokens, layer, pos, metric
        )
        # Normalize: 0 = completely corrupted, 1 = perfectly restored to clean run
        normalized_score = (patched_val - corrupted_baseline) / (clean_baseline - corrupted_baseline)
        patching_matrix[layer, pos] = normalized_score

# 5. Render results
str_tokens = model.to_str_tokens(clean_tokens[0]) # Use first prompt tokens as baseline labels
fig = plot_patching_heatmap(patching_matrix, str_tokens, title="Geographical Fact Extraction Circuit Map")
fig.show()