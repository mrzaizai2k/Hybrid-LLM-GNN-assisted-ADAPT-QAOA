import sys
sys.path.append("")

from datetime import datetime
import numpy as np

import os
import glob
import json
import time
from typing import Dict, Tuple, List, Optional, Any

import pandas as pd
import networkx as nx

from qaoa import QAOA, problems, mixers, initialstates
from qiskit_algorithms.optimizers import L_BFGS_B

from src.utils import maxcut_bruteforce


# =========================
# File / Data Utilities
# =========================
def find_graph_csv(graph_dir: str) -> str:
    csv_files = glob.glob(os.path.join(graph_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {graph_dir}")

    csv_files.sort()
    return csv_files[-1]


def load_graphs_from_csv(
    csv_path: str,
    n_samples: Optional[int] = None
) -> Tuple[Dict[str, nx.Graph], pd.DataFrame]:

    df = pd.read_csv(csv_path)

    if n_samples is not None:
        df = df.head(n_samples)

    graphs: Dict[str, nx.Graph] = {}

    for _, row in df.iterrows():
        graph_id = f"graph_{int(row['graph_num'])}"
        edgelist = json.loads(row["edgelist_json"])

        G = nx.Graph()
        for u, v, w in edgelist:
            G.add_edge(int(u), int(v), weight=float(w))

        graphs[graph_id] = G

    return graphs, df


def graph_to_edgelist(G: nx.Graph) -> List[List[float]]:
    return [[u, v, d.get("weight", 1.0)] for u, v, d in G.edges(data=True)]


def infer_n_nodes(graphs: Dict[str, nx.Graph]) -> int:
    return next(iter(graphs.values())).number_of_nodes()


# =========================
# QAOA Core
# =========================
def run_qaoa_on_graph(
    graph_name: str,
    G: nx.Graph,
    depth: int
) -> Dict[str, Any]:

    start = time.time()

    qaoa = QAOA(
        problem=problems.MaxCut(G),
        mixer=mixers.X(),
        initialstate=initialstates.Plus(),
        interpolate=True,
        optimizer=[L_BFGS_B, {
            "maxiter": 50,
            "ftol": 1e-9
        }]
    )

    if depth == 1:
        qaoa.sample_cost_landscape()

    qaoa.optimize(depth=depth)

    exp_val = qaoa.get_Exp(depth=depth)
    var_val = qaoa.get_Var(depth=depth)

    gamma = qaoa.get_gamma(depth=depth)
    beta = qaoa.get_beta(depth=depth)

    # Convert to clean Python lists (fix core bug)
    if isinstance(gamma, np.ndarray):
        gamma = gamma.tolist()
    if isinstance(beta, np.ndarray):
        beta = beta.tolist()

    energy_opt, _ = maxcut_bruteforce(G)
    approx_ratio = exp_val / energy_opt if energy_opt != 0 else None

    end = time.time()

    return {
        "graph_name": graph_name,
        "n_nodes": G.number_of_nodes(),
        "edgelist_list_len": G.number_of_edges(),
        "n_layers": depth,
        "expected_energy": exp_val,
        "variance": var_val,
        "γ_coeff": gamma,
        "β_coeff": beta,
        "approx_ratio": approx_ratio,
        "energy_mqlib": energy_opt,
        "edgelist_list": graph_to_edgelist(G),
        "took_time": round(end - start, 3),
        "method": "vanilla_qaoa",
        "optimizer": "BFGS"
    }


# =========================
# Incremental Saving
# =========================
def append_result_to_csv(
    result: Dict[str, Any],
    output_path: str
) -> None:

    # Ensure list fields are saved correctly
    result["γ_coeff"] = json.dumps(result["γ_coeff"])
    result["β_coeff"] = json.dumps(result["β_coeff"])
    result["edgelist_list"] = json.dumps(result["edgelist_list"])

    df = pd.DataFrame([result])

    write_header = not os.path.exists(output_path)

    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False
    )


# =========================
# Experiment (UPDATED)
# =========================
def run_experiment_streaming(
    graphs: Dict[str, nx.Graph],
    depth: int,
    n_runs: int,
    output_path: str
) -> None:
    """
    Run QAOA and save results incrementally after each graph.

    This prevents data loss for long experiments.
    """
    for run_idx in range(n_runs):
        print(f"\n=== RUN {run_idx + 1}/{n_runs} ===")

        for graph_name, G in graphs.items():
            print(f"Running {graph_name}...")

            result = run_qaoa_on_graph(graph_name, G, depth)
            result["run_id"] = run_idx

            append_result_to_csv(result, output_path)

            print(f"Saved result for {graph_name}")


# =========================
# Public API
# =========================
def run_vanilla_qaoa(
    data_path: str,
    depth: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_runs: int = 1,
    overwrite: bool = True
) -> pd.DataFrame:
    """
    Run QAOA with incremental saving.

    Args:
        data_path: Path to dataset
        depth: QAOA depth (None = n_nodes)
        n_samples: Number of graphs to load
        n_runs: Number of runs
        overwrite: If True, delete old CSV before running

    Returns:
        Final DataFrame
    """
    graph_dir = os.path.join(data_path, "graphs")
    output_dir = os.path.join(data_path, "qaoa_result")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%d_%H_%M")
    output_filename = f"qaoa_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # --- overwrite logic
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
        print("Old result file removed.")

    # --- load graphs
    csv_path = find_graph_csv(graph_dir)
    graphs, _ = load_graphs_from_csv(csv_path, n_samples=n_samples)

    if not graphs:
        raise ValueError("No graphs loaded.")

    # --- auto depth
    if depth is None:
        depth = infer_n_nodes(graphs)

    # --- run streaming experiment
    run_experiment_streaming(graphs, depth, n_runs, output_path)

    # --- load final result
    df = pd.read_csv(output_path)
    return df


# =========================
# CLI
# =========================
if __name__ == "__main__":
    df = run_vanilla_qaoa(
        data_path="ADAPT.jl_results/test/time/graphs_n4",
        depth=None,
        n_samples=2,
        n_runs=1
    )

    print(df.head())
    