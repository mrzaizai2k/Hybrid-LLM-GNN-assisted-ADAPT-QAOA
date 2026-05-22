# %% [markdown]
# # Imports

# %%
import subprocess
import os
import re
from datetime import datetime
from pathlib import Path
import json
from itertools import islice
import pandas as pd
import networkx as nx
import random
from src.vanilla_qaoa_result import run_vanilla_qaoa

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %%
from src.adapt_utils import (
    run_adapt_jl_parallel,
    show_adapt_logs,
    get_combined_res_df
)

# %% [markdown]
# # Config

# %%
adapt_gpt_dir = Path(
    "/home/mrzaizai2k/code_bao/ADAPT_GPT"
)
adapt_output_dir = "./ADAPT.jl_results/test/time"
n_graphs = 5
n_runs = 1
input_graph_filename = "ADAPT.jl_results/graphs.json"


# %%
def add_weights_to_nx_graph(G, weighted=True, use_negative=False):
    elist = []
    for u, v in G.edges():
        if weighted:
            w = random.uniform(0.1, 1.0)
            if use_negative and random.random() < 0.5:
                w *= -1
        else:
            w = -1 if (use_negative and random.random() < 0.5) else 1

        elist.append([int(u)+1, int(v)+1, float(round(w, 2))])
        # +1 to match Julia 1-indexing
    return elist

def generate_graphs(
    n_graphs=10,
    n_nodes=10,
    density=None,          # if None → random
    weighted=True,
    use_negative=False
):
    graphs_dict = {}

    for i in range(n_graphs):
        if density is None:
            p = random.uniform(0.6, 0.9)   # random density
        else:
            p = density

        G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        # avoid empty graph
        while G.number_of_edges() == 0:
            G = nx.erdos_renyi_graph(n=n_nodes, p=p)

        elist = add_weights_to_nx_graph(G, weighted, use_negative)

        graph_name = f"Graph_{i}_n{n_nodes}"

        graphs_dict[graph_name] = {
            "elist": elist,
            "n_nodes": n_nodes
        }

    return graphs_dict

def load_graphs(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_graphs_to_json(graphs_dict, filename):
    with open(filename, "w") as f:
        json.dump(graphs_dict, f, indent=2)

# %% [markdown]
# # Generate graphs

# %%
path_list = []
for i in [4, 6, 8, 10, 11]:
    graphs = generate_graphs(
        n_graphs=n_graphs,
        n_nodes=i,
        density=None,          # or e.g. 0.7
        weighted=True,
        use_negative=False
    )
    path = f"{adapt_output_dir}/graphs_n{i}.json"

    save_graphs_to_json(graphs, path)
    path_list.append(path)

# Load back
cur_input_graphs_dict = load_graphs(path)


# %% [markdown]
# # ADAPT QAOA result

# %%
adapt_folder_list = []
for path in path_list:
    filename = os.path.basename(path)
    match = re.search(r"graphs_n(\d+)", filename)

    if not match:
        raise ValueError(f"Cannot extract node number from {filename}")

    i = int(match.group(1))  # convert to int

    new_name = f"{adapt_output_dir}/graphs_n{i}/"
    adapt_folder_list.append(new_name)

    logs_list, cur_proc = run_adapt_jl_parallel(
        script_dir=adapt_gpt_dir,
        output_dir=new_name,
        input_graphs=path,
        n_workers=1,
        graphs_number=n_graphs,
        n_nodes=i,
        trials_per_graph=n_runs,
        max_params=50,
        gamma_0="gamma0_grid.json",
        pool_name="qaoa_double_pool",
        use_floor_stopper=True,
        temp_folder=f"{adapt_output_dir}/temp_data_n{i}",
    )

# %%
show_adapt_logs(logs_list, n_lines=20)

# %% [markdown]
# # QAOA result

# %%
adapt_folder_list

# %%
for path in adapt_folder_list:
    qaoa_df = run_vanilla_qaoa(
        data_path=path,
        depth=None,
        n_samples=None,
        n_runs=1
    )

# %%
qaoa_df.head()

# %% [markdown]
# # LLMs

# %%
# ------------------------
# IMPORTS
# ------------------------
import time
import numpy as np
import torch
import glob
from src.adapt_utils import compute_metrics_per_graph
from src.model_interface import QAOA_GPT
from src.utils import (
    attach_resolved_names,
    load_and_aggregate_adapt,
    build_results_df,
    build_final_df,
    build_summary_df,
)
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

# %%
# ── Model Configs ────────────────────────────────────────────
# Each entry: ckpt, data_dir, and optionally name.
# Auto-extraction from ckpt filename: arch (element[0]) + method (element[3])
# e.g. "llama_ckpt_5000_feather_ar_0_89184__er_0_0.pt" → "LLaMA-Feather"
# Provide `name` explicitly to override auto-extraction.

MODEL_CONFIGS = [
    dict(
        ckpt="nanoGPT/out-11_nodes_feather/llama_ckpt_5000_feather_ar_0_89184__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_feather",
    ),
    dict(
        ckpt="nanoGPT/out-11_nodes_gnn/llama_ckpt_4500_gnn_ar_0_91473__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_gnn",
    ),
    dict(
        ckpt="nanoGPT/out-11_nodes_netlsd/llama_ckpt_5000_netlsd_ar_0_9264__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_netlsd",
    ),
    dict(
        ckpt="nanoGPT/out-11_nodes_feather/gpt_ckpt_3000_feather_ar_0_96249__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_feather",
    ),
    dict(
        ckpt="nanoGPT/out-11_nodes_gnn/gpt_ckpt_2500_gnn_ar_0_95371__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_gnn",
    ),
    dict(
        ckpt="nanoGPT/out-11_nodes_netlsd/gpt_ckpt_4000_netlsd_ar_0_96605__er_0_0.pt",
        data_dir="nanoGPT/data/11_nodes_netlsd",
    ),
]

# %%
# ── Config ──────────────────────────────────────────────────
SEED            = 1337
BASE_DIR        = adapt_output_dir           # reuse from above
MAX_TOKENS      = 200
LLM_TEMPERATURE = 0.1
LLM_TOP_K       = 200
N_SAMPLES       = 5
# ───────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CONFIGS = attach_resolved_names(MODEL_CONFIGS)
print(f"MODEL_CONFIGS: {MODEL_CONFIGS}")


# %% [markdown]
# # Helper functions

# %%
def get_graph_folders(base_dir: str) -> list[str]:
    """Return all graphs_nXX sub-folders (directories only), sorted by node count."""
    pattern = os.path.join(base_dir, "graphs_n*")
    folders = sorted([
        p for p in glob.glob(pattern)
        if os.path.isdir(p)
    ])
    return folders


def qaoa_time_mean(folder: str) -> tuple[float, int]:
    qaoa_dir = os.path.join(folder, "qaoa_result")
    matches = glob.glob(os.path.join(qaoa_dir, "qaoa_*.csv"))

    if not matches:
        raise FileNotFoundError(f"No qaoa_*.csv found in {qaoa_dir}")

    csv_path = matches[0]
    print(f"        Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    n_runs = df["run_id"].nunique()
    per_graph = df.groupby("graph_name")["took_time"].mean()
    grand_mean = per_graph.mean()

    return grand_mean, n_runs


def adapt_time_mean(folder: str) -> tuple[float, int]:
    _, adapt_agg, _, _ = load_and_aggregate_adapt(folder)

    mean_time = adapt_agg["adapt_time_mean"].mean()
    n_runs = int(adapt_agg["adapt_n_runs"].mode()[0])

    return mean_time, n_runs


def llm_time_mean(
    folder: str,
    graphs_unique,
    n_runs: int,
    n_nodes: int,
    model_configs: list,
) -> dict:
    """
    Run each model in model_configs on graphs_unique and return a dict
    mapping resolved_name → mean time per run.
    """
    times = {}
    for cfg in model_configs:
        model_name = cfg["resolved_name"]
        model = QAOA_GPT(model_ckpt=cfg["ckpt"], data_dir=cfg["data_dir"])
        model.n_nodes = n_nodes

        t0 = time.perf_counter()
        df_model = model.generate_circ_from_nx(
            graphs_unique,
            num_samples=n_runs,
            max_new_tokens=MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_k=LLM_TOP_K,
        )
        total_wall = time.perf_counter() - t0

        if "took_time" in df_model.columns:
            per_run = df_model["took_time"].sum() / (len(df_model) * n_runs)
        else:
            per_run = total_wall / (len(graphs_unique) * n_runs)

        times[model_name] = per_run
    return times


def llm_ar_mean(
    graphs_unique,
    n_nodes: int,
    model_configs: list,
) -> dict:
    """
    Run each model in model_configs, evaluate circuits, and return a dict
    mapping resolved_name → mean AR across graphs.
    """
    ar_results = {}
    for cfg in model_configs:
        model_name = cfg["resolved_name"]
        model = QAOA_GPT(model_ckpt=cfg["ckpt"], data_dir=cfg["data_dir"])
        model.n_nodes = n_nodes

        df_model = model.generate_circ_from_nx(
            graphs_unique,
            num_samples=N_SAMPLES,
            max_new_tokens=MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_k=LLM_TOP_K,
        )
        df_eval = model.eval_circ_df_jl(df_model)
        ar, layers, error_rate = compute_metrics_per_graph(df_eval)

        # ar is per-graph; take the mean across graphs
        ar_results[model_name] = float(np.mean(list(ar.values())))
    return ar_results


# %% [markdown]
# # Build timing + AR summary

# %%
def build_timing_ar_summary(model_configs: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    timing_df : columns [folder, n_nodes, n_runs, qaoa_time_mean_s,
                          adapt_time_mean_s, <model>_time_mean_s, ...]
    ar_df     : columns [folder, n_nodes, qaoa_ar_mean, adapt_ar_mean,
                          <model>_ar_mean, ...]
    """
    folders = get_graph_folders(BASE_DIR)
    if not folders:
        raise FileNotFoundError(f"No graphs_nXX folders found under {BASE_DIR}")

    timing_records = []
    ar_records = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        n_nodes = int(folder_name.replace("graphs_n", ""))

        print(f"\n{'='*55}")
        print(f"Processing {folder_name}  (n_nodes={n_nodes})")
        print(f"{'='*55}")

        # ── 1. QAOA timing ──────────────────────────────────
        print("  [1/4] Reading QAOA times …")
        try:
            qaoa_mean_time, n_runs_qaoa = qaoa_time_mean(folder)
            print(f"        QAOA mean time/run = {qaoa_mean_time:.4f}s  (n_runs={n_runs_qaoa})")
        except FileNotFoundError:
            print("        qaoa_result not found – skipping QAOA.")
            qaoa_mean_time, n_runs_qaoa = np.nan, np.nan

        # ── 2. ADAPT timing ─────────────────────────────────
        print("  [2/4] Loading ADAPT aggregation …")
        try:
            adapt_mean_time, n_runs_adapt = adapt_time_mean(folder)
            print(f"        ADAPT mean time/run = {adapt_mean_time:.4f}s  (n_runs={n_runs_adapt})")
        except Exception as e:
            print(f"        ADAPT load failed: {e}")
            adapt_mean_time, n_runs_adapt = np.nan, np.nan

        n_runs = int(n_runs_adapt if not np.isnan(n_runs_adapt) else n_runs_qaoa)

        # Load graphs + adapt aggregation for AR
        _, adapt_agg, graphs_unique, meta_df = load_and_aggregate_adapt(folder)

        # ── 3. LLM timing ───────────────────────────────────
        print(f"  [3/4] Running LLM inference for timing  (n_runs={n_runs}) …")
        try:
            llm_times = llm_time_mean(folder, graphs_unique, n_runs, n_nodes, model_configs)
            for name, t in llm_times.items():
                print(f"        {name} mean time/run = {t:.4f}s")
        except Exception as e:
            print(f"        LLM timing failed: {e}")
            llm_times = {cfg["resolved_name"]: np.nan for cfg in model_configs}

        # ── 4. AR comparison ────────────────────────────────
        print(f"  [4/4] Computing AR for all methods …")

        # QAOA AR
        try:
            qaoa_csv = glob.glob(os.path.join(folder, "qaoa_result", "qaoa_*.csv"))[0]
            qaoa_res = pd.read_csv(qaoa_csv)
            qaoa_ar_mean = qaoa_res.groupby("graph_name")["approximation_ratio"].mean().mean()
            print(f"        QAOA mean AR = {qaoa_ar_mean:.4f}")
        except Exception as e:
            print(f"        QAOA AR failed: {e}")
            qaoa_ar_mean = np.nan

        # ADAPT AR
        try:
            adapt_ar_mean = adapt_agg["adapt_ar_mean"].mean()
            print(f"        ADAPT mean AR = {adapt_ar_mean:.4f}")
        except Exception as e:
            print(f"        ADAPT AR failed: {e}")
            adapt_ar_mean = np.nan

        # LLM AR
        try:
            llm_ars = llm_ar_mean(graphs_unique, n_nodes, model_configs)
            for name, ar in llm_ars.items():
                print(f"        {name} mean AR = {ar:.4f}")
        except Exception as e:
            print(f"        LLM AR failed: {e}")
            llm_ars = {cfg["resolved_name"]: np.nan for cfg in model_configs}

        # Build records
        timing_row = {
            "folder":            folder_name,
            "n_nodes":           n_nodes,
            "n_runs":            n_runs,
            "qaoa_time_mean_s":  round(qaoa_mean_time,  6),
            "adapt_time_mean_s": round(adapt_mean_time, 6),
        }
        timing_row.update({f"{k}_time_mean_s": round(v, 6) for k, v in llm_times.items()})
        timing_records.append(timing_row)

        ar_row = {
            "folder":       folder_name,
            "n_nodes":      n_nodes,
            "qaoa_ar_mean": round(qaoa_ar_mean,  6),
            "adapt_ar_mean": round(adapt_ar_mean, 6),
        }
        ar_row.update({f"{k}_ar_mean": round(v, 6) for k, v in llm_ars.items()})
        ar_records.append(ar_row)

    timing_df = (
        pd.DataFrame(timing_records)
        .sort_values("n_nodes")
        .reset_index(drop=True)
    )
    ar_df = (
        pd.DataFrame(ar_records)
        .sort_values("n_nodes")
        .reset_index(drop=True)
    )
    return timing_df, ar_df


# %%
timing_summary, ar_summary = build_timing_ar_summary(MODEL_CONFIGS)

print("\n=== Timing Summary ===")
print(timing_summary.to_string(index=False))

print("\n=== AR Summary ===")
print(ar_summary.to_string(index=False))

# %%
timing_summary.to_csv(f"{BASE_DIR}/timing_summary.csv", index=False)
ar_summary.to_csv(f"{BASE_DIR}/ar_summary.csv", index=False)

# %% [markdown]
# # Plots

# %%
def plot_timing_comparison(summary_df: pd.DataFrame, save_path: str = None, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Fixed methods
    fixed_methods = {
        "QAOA":  ("qaoa_time_mean_s",  "o-",  "#E74C3C"),
        "ADAPT": ("adapt_time_mean_s", "s--", "#3498DB"),
    }

    # LLM model columns (everything else ending in _time_mean_s)
    llm_cols = [c for c in summary_df.columns
                if c.endswith("_time_mean_s")
                and c not in ("qaoa_time_mean_s", "adapt_time_mean_s")]

    colors_llm = plt.cm.tab10.colors

    for label, (col, style, color) in fixed_methods.items():
        ax.plot(
            summary_df["n_nodes"], summary_df[col],
            style, color=color, linewidth=2, markersize=8, label=label,
        )

    for idx, col in enumerate(llm_cols):
        label = col.replace("_time_mean_s", "")
        ax.plot(
            summary_df["n_nodes"], summary_df[col],
            "^:", color=colors_llm[idx % len(colors_llm)],
            linewidth=2, markersize=8, label=label,
        )

    ax.set_title("Mean Inference Time per Method vs Graph Size", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Nodes", fontsize=13)
    ax.set_xticks(summary_df["n_nodes"])
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Mean Time per Run (s) — log scale", fontsize=13)
    else:
        ax.set_ylabel("Mean Time per Run (s)", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


def plot_ar_comparison(ar_df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 8))

    fixed_methods = {
        "QAOA":  ("qaoa_ar_mean",  "o-",  "#E74C3C"),
        "ADAPT": ("adapt_ar_mean", "s--", "#3498DB"),
    }

    llm_cols = [c for c in ar_df.columns
                if c.endswith("_ar_mean")
                and c not in ("qaoa_ar_mean", "adapt_ar_mean")]

    colors_llm = plt.cm.tab10.colors

    for label, (col, style, color) in fixed_methods.items():
        ax.plot(
            ar_df["n_nodes"], ar_df[col],
            style, color=color, linewidth=2, markersize=8, label=label,
        )

    for idx, col in enumerate(llm_cols):
        label = col.replace("_ar_mean", "")
        ax.plot(
            ar_df["n_nodes"], ar_df[col],
            "^:", color=colors_llm[idx % len(colors_llm)],
            linewidth=2, markersize=8, label=label,
        )

    ax.set_title("Mean Approximation Ratio per Method vs Graph Size", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Nodes", fontsize=13)
    ax.set_ylabel("Mean AR", fontsize=13)
    ax.set_xticks(ar_df["n_nodes"])
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()


# %%
plot_timing_comparison(timing_summary, save_path=f"{BASE_DIR}/timing_comparison.png")
plot_timing_comparison(timing_summary, save_path=f"{BASE_DIR}/timing_comparison_log.png", log_scale=True)

# %%
plot_ar_comparison(ar_summary, save_path=f"{BASE_DIR}/ar_comparison.png")