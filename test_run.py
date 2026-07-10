# Save this temporarily as test_run.py in your main folder
import json
import torch
from transformer_lens import HookedTransformer
from src.patching import compute_logit_diff, patch_residual_stream_layer

# 1. Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Mock a clean and corrupted data point
clean_tokens = model.to_tokens(["The capital of France is"])
corrupted_tokens = model.to_tokens(["The capital of Japan is"])
correct_ids = torch.tensor([model.to_single_token(" Paris")], device=device)
incorrect_ids = torch.tensor([model.to_single_token(" Tokyo")], device=device)

# 3. Test baselines
clean_logits = model(clean_tokens)
corrupted_logits = model(corrupted_tokens)

metric = lambda logits: compute_logit_diff(logits, correct_ids, incorrect_ids)
print(f"Clean Baseline Logit Diff: {metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline Logit Diff: {metric(corrupted_logits).item():.4f}")

# 4. Test a single hook intervention (Layer 5, Position 3)
_, clean_cache = model.run_with_cache(clean_tokens)
patched_score = patch_residual_stream_layer(model, clean_cache, corrupted_tokens, target_layer=5, target_pos=3, metric_fn=metric)
print(f"Single Patch Test Score: {patched_score:.4f}")
print("System Status: ALL CORE MECHANICS OPERATIONAL!")