# Hybrid LLM-GNN Assisted ADAPT-QAOA for Optimizing Graph-Structured Quantum Circuits

## 📑 Table of Contents

* [Overview](#overview)
* [Key Contributions](#key-contributions)
* [Quick Start Guide](#quick-start-guide)

  * [Installation](#installation)
  * [Pipeline](#pipeline)
* [Usage](#usage)
* [Data Availability](#data-availability)
* [Project Structure](#project-structure)

---

## Overview

Combinatorial optimization problems are central to applications such as logistics and network design, yet they become increasingly difficult for classical algorithms as problem size grows.

Hybrid quantum–classical methods like **QAOA (Quantum Approximate Optimization Algorithm)** offer a promising alternative. Its adaptive variant improves flexibility by dynamically constructing circuits, but still faces two major challenges:

* Efficient circuit structure generation
* Stable and effective parameter initialization

Existing approaches typically treat these problems separately, limiting generalization and performance—especially on weighted graphs.

This project introduces a **unified generative–predictive framework** that:

* Encodes graph structure via embeddings (**NetLSD, FEATHER, GNNs**)
* Uses transformer models (**nanoGPT, LLaMA**) to autoregressively generate circuits
* Jointly models circuit structure and parameters in a single sequence

The result is:

* Faster convergence
* Improved approximation quality
* Reduced circuit depth with maintained performance

---

## Key Contributions

* Unified framework combining **generation + optimization**
* Integration of **graph embeddings into LLMs**
* Support for multiple embedding methods:

  * GNN
  * NetLSD
  * FEATHER
* Improved generalization across:

  * Circuit depths
  * Graph types (weighted & unweighted)

---

## Quick Start Guide

### Installation

#### 1. Julia Setup

1. Install Julia: [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. (Optional) Add Jupyter kernel: [https://julialang.github.io/IJulia.jl/stable/manual/installation/](https://julialang.github.io/IJulia.jl/stable/manual/installation/)

You should check this code `install_julia.md` for more

#### 2. Python Setup

```bash
conda create -n adapt_gpt python=3.10 -y
conda activate adapt_gpt
pip install -r requirements.txt
```

#### 3. Clone Repository

Run Julia and install dependencies:

1. Clone this repo with its dependencies: `git clone https://github.com/IlyaTyagin/ADAPT-GPT --recurse-submodules`
2. `cd ADAPT-GPT/ADAPT.jl/`
3. Run julia: `julia --project=.`
4. Install Julia dependencies. Inside julia interpreter run: `julia> using Pkg; Pkg.instantiate(); Pkg.add(["JuMP", "MQLib" , "ProgressBars", "SimpleWeightedGraphs", "CSV", "DataFrames", "JSON", "ArgParse", "Multibreak"]); Pkg.develop(path="SciPyOptimizers");` 


---

### Pipeline

#### 1. Generate Graph–Circuit Data

```bash
./adapt_maxcut_run_multithread.sh
```

⚠️ Recommended: ≥ 50k circuits (requires CPU cluster)

---

#### 2. (Optional) Train GNN Embeddings

* Configure: `config/config.yaml`
* Run: `gnn_training.ipynb`
* Save models to: `models/`

---

#### 3. Prepare Dataset for LLM

```bash
python prepare_circ.py --adapt_results_dir <ADAPT_RESULTS_DIR> --save_dir <SAVING_DIR> --n_nodes <N_NODES> --embedding_method <embedding_method>
```
where `<N_NODES>` is the problem size (qubits/graph nodes) for all circuits in the dataset, embedding methods are `'feather', 'gnn', 'netlsd'`

📌 Note: `<SAVING_DIR>` must be inside `nanoGPT/data`

---

#### 4. Train LLM (nanoGPT / LLaMA)

```bash
cd nanoGPT/

python train_pad_gemb_ar_eval.py --train_config_path <SAVING_DIR>/train_adapt_gpt_config.py --model <gpt | llama>
```

---

#### 5. Inference

* Notebook: `qaoa_gpt_inference_demo.ipynb`
* Generates circuits and evaluates with ADAPT.jl

---

## Usage

* Example commands available in `Makefile`
* Supports:

  * Circuit generation
  * Evaluation vs ADAPT-QAOA & vanilla QAOA
  * Embedding comparison
  * Model architecture comparison

---

## Data Availability

* Pre-trained models:
  👉 [https://drive.google.com/drive/folders/1ddMW1iLYlhd_Nb-ZyRFY1ktdZ9tDlNjQ](https://drive.google.com/drive/folders/1ddMW1iLYlhd_Nb-ZyRFY1ktdZ9tDlNjQ)

Notes:

* No precomputed embedding datasets are provided
* Users are encouraged to generate embeddings independently

⚠️ Important:

* `adapt_maxcut_run_multithread.sh` does **not filter low AR circuits**
* Filtering is handled in `prepare_circ.py`

---

## Project Structure

```
ADAPT.jl_results/              # Generated training data
ADAPT.jl_results/test/         # Evaluation datasets

config/config.yaml             # Embedding configuration

prepare_circ.py                # Data preprocessing & tokenization

src/
├── embedding/                 # GNN, NetLSD, FEATHER implementations
├── adapt_utils.py             # ADAPT result utilities
├── circuit_util.py            # Input preparation
├── model_interface.py         # LLM inference interface
├── utils.py                   # Common utilities & plotting
├── vanilla_qaoa_result.py     # Baseline QAOA evaluation

models/                        # Trained GNN models

nanoGPT/
├── data/                      # Processed datasets
├── model_llama.py             # LLaMA implementation
├── model_pad_gemb.py          # nanoGPT variant
├── train_pad_gemb_ar_eval.py  # Training script

notebooks/
├── gnn_training.ipynb
├── emb_comparison.ipynb
├── Adapt_llm_comparision.ipynb
├── ar_emb_compare.ipynb
├── qaoa_gpt_inference_demo.ipynb

scripts/
└── adapt_maxcut_run_multithread.sh

evaluation/
└── adapt_gpt_eval_energy.jl

docs/                          # Visualization results (n = 9, 10, 11)
```

