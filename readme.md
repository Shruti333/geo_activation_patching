# Causal Localization of Fact Retrieval Circuits in GPT-2 Small

This repository contains a clean, production-grade implementation of **Exploratory Activation Patching** (causal mediation analysis) using `TransformerLens`. The goal is to reverse-engineer and map the specific token-layer bottlenecks through which a transformer retrieves geographical facts (Country $\rightarrow$ Capital).
# Causal Localization of Fact Retrieval Circuits in GPT-2 Small

This repository contains a clean, production-grade implementation of **Exploratory Activation Patching** (causal mediation analysis) using `TransformerLens`. The goal is to reverse-engineer and map the specific token-layer bottlenecks through which a small transformer model (`gpt2-small`) retrieves and routes geographical facts (Country $\rightarrow$ Capital).

## Project Overview

Mechanistic Interpretability seeks to move past treating neural networks as uninterpretable "black boxes" by reverse-engineering internal weights into discrete, human-understandable circuits. 

This project implements **Denoising Activation Patching**. By setting up a counterfactual pair of prompts with identical token lengths, we can corrupt the network's internal processing and surgically inject "clean" cached activations layer-by-layer and token-by-token. This allows us to map exactly where the model isolates factual knowledge and how it routes that knowledge to predict the next token.

---

## Repository Structure

```text
my_project/
├── data/
│   └── prompts.json          # Batch evaluation prompts (JSON dataset)
├── src/
│   ├── __init__.py           # Makes src folder importable
│   ├── patching.py           # Core tensor manipulation and hooking logic
│   └── visualize.py          # Publication-ready Plotly configurations
├── main.py                   # Production pipeline CLI execution script
├── requirements.txt          # Python dependencies
└── .gitignore                # Git filtering exclusions

## Project Setup & Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
## Results & Technical Analysis

The generated 2D activation patching heatmap provides empirical proof of how factual knowledge is localized and processed within `gpt2-small`. By tracking the normalized logit difference recovery, we map a distinct two-stage circuit:

### 1. The Information Storage Bottleneck (Layers 0–9)
* **Observation:** A massive, dark blue block of near-perfect recovery ($\sim 1.0$) stays tightly bound to the subject token position (`" France"`) across the early and mid-layers.
* **Mechanism:** This demonstrates that the transformer dedicated its early processing layers to localized factual lookup. The model retrieves the attributes of the country noun and stores that semantic vector directly within the residual stream *at the physical position of that word*. 

### 2. The Late-Stage Routing Circuit (Layers 10–11)
* **Observation:** Between Layer 9 and Layer 10, the causal influence sharply drops from the `" France"` column and simultaneously lights up on the final token position (`" is"`).
* **Mechanism:** Because autoregressive language models predict the next token using only the hidden states of the *final prompt position*, the information cannot remain on the subject token. This sudden shift captures **Information Mover Attention Heads** in action. These late-stage heads read the factual vector from the country slot and copy it over to the end of the context window, allowing the unembedding layer to output the logit for ` Paris`.