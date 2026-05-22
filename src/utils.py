
import time
import yaml
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import re
from typing import Tuple, List, Dict
 

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper



def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def add_weights_to_nx_graph(nx_graph):
    for u, v in nx_graph.edges():
        w = round(random.uniform(0, 1), 2)
        while w == 0:
            w = round(random.uniform(0, 1), 2)
        nx_graph[u][v]["weight"] = w
    return nx_graph


def generate_er_graphs(n_graphs, n_nodes):

    graphs = {}

    for i in range(n_graphs):

        p = random.randrange(6, 9) / 10

        g = nx.erdos_renyi_graph(
            n=n_nodes,
            p=p
        )

        g = add_weights_to_nx_graph(g)

        graphs[f"er_graph_{i}"] = g

    return graphs



 
ARCH_ALIASES: Dict[str, str] = {
    "llama": "LLaMA",
    "gpt":   "NanoGPT",
}
 
METHOD_ALIASES: Dict[str, str] = {
    "netlsd":  "NetLSD",
    "feather": "Feather",
    "gnn":     "GNN",
}
 
# ---------------------------------------------------------------------------
# NAME EXTRACTION
# ---------------------------------------------------------------------------
 
def extract_arch(ckpt_path: str) -> str:
    """
    Extract the architecture token from a checkpoint filename.
 
    The filename is expected to follow the pattern:
        <arch>_ckpt_<step>_<method>_...
 
    Examples:
        "llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt" -> "LLaMA"
        "gpt_ckpt_3500_feather_ar_0_957__er_0_0.pt" -> "GPT"
 
    Returns the alias from ARCH_ALIASES if found, otherwise the raw token uppercased.
    """
    basename = ckpt_path.split("/")[-1]
    parts    = basename.split("_")
    try:
        raw = parts[0].lower()
        return ARCH_ALIASES.get(raw, raw.upper())
    except IndexError:
        return basename
 
 
def extract_method(ckpt_path: str) -> str:
    """
    Extract the embedding/method token from a checkpoint filename.
 
    The filename is expected to follow the pattern:
        <arch>_ckpt_<step>_<method>_...
 
    Examples:
        "llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt" -> "GNN"
        "gpt_ckpt_3500_netlsd_ar_0_957__er_0_0.pt"  -> "NetLSD"
 
    Returns the alias from METHOD_ALIASES if found, otherwise the raw token uppercased.
    """
    basename = ckpt_path.split("/")[-1]
    parts    = basename.split("_")
    try:
        raw = parts[3].lower()
        return METHOD_ALIASES.get(raw, raw.upper())
    except IndexError:
        return "UNKNOWN"
 
 
def extract_model_name(ckpt_path: str) -> str:
    """
    Auto-extract a human-readable "<Arch>-<Method>" name from a checkpoint path.
 
    Examples:
        "nanoGPT/out-10_nodes_gnn/llama_ckpt_5500_gnn_ar_0_924__er_0_006.pt"
            -> "LLaMA-GNN"
        "nanoGPT/out-10_nodes_feather/gpt_ckpt_3500_feather_ar_0_957__er_0_0.pt"
            -> "GPT-Feather"
 
    Falls back to the raw basename if parsing fails.
    """
    try:
        arch   = extract_arch(ckpt_path)
        method = extract_method(ckpt_path)
        return f"{arch}-{method}"
    except Exception:
        return ckpt_path.split("/")[-1]
 
 
def resolve_model_name(cfg: dict) -> str:
    """
    Return cfg['name'] if explicitly set, otherwise auto-extract from cfg['ckpt'].
 
    This allows per-entry overrides:
        dict(name="My Custom Name", ckpt="...", data_dir="...")  -> "My Custom Name"
        dict(ckpt="llama_ckpt_5500_gnn...", data_dir="...")      -> "LLaMA-GNN"
    """
    return cfg.get("name") or extract_model_name(cfg["ckpt"])
 
 
def attach_resolved_names(model_configs: List[dict]) -> List[dict]:
    """
    Attach 'resolved_name', 'arch', and 'method' keys to every config in-place.
    Also prints a summary of resolved names.
    """
    for cfg in model_configs:
        cfg["resolved_name"] = resolve_model_name(cfg)
        cfg["arch"]          = extract_arch(cfg["ckpt"])
        cfg["method"]        = extract_method(cfg["ckpt"])
 
    print("Resolved model names:")
    for cfg in model_configs:
        print(f"  {cfg['resolved_name']}  (arch={cfg['arch']}, method={cfg['method']})")
 
    return model_configs
 
# ---------------------------------------------------------------------------
# GRAPH UTILITIES
# ---------------------------------------------------------------------------
 
def graph_name_to_num(graph_name: str) -> int:
    """
    Extract the trailing integer from a graph_name string.
 
    Examples:
        'graph_007'  -> 7
        'g42'        -> 42
        'graph_0042' -> 42
 
    Falls back to 0 if no number is found.
    """
    match = re.search(r"(\d+)$", str(graph_name))
    return int(match.group(1)) if match else 0
 
 
def edgelist_to_nx(edgelist, n_nodes: int) -> nx.Graph:
    """Convert a list of (u, v, w) 1-indexed edge tuples to a NetworkX Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in edgelist:
        G.add_edge(u - 1, v - 1, weight=w)
    return G
 
 
def load_graphs_from_adapt(adapt_df: pd.DataFrame) -> Tuple[List[nx.Graph], pd.DataFrame]:
    """
    Build NetworkX graphs from a (pre-deduplicated) ADAPT DataFrame.
 
    Returns:
        graphs  : list of nx.Graph, one per row
        meta_df : DataFrame with columns ['graph_name', 'graph_num']
    """
    graphs, meta = [], []
    for _, row in adapt_df.iterrows():
        G = edgelist_to_nx(row["edgelist_list"], row["n_nodes"])
        graphs.append(G)
        meta.append({
            "graph_name": row["graph_name"],
            "graph_num":  graph_name_to_num(row["graph_name"]),
        })
    return graphs, pd.DataFrame(meta)
 
# ---------------------------------------------------------------------------
# ADAPT AGGREGATION
# ---------------------------------------------------------------------------
def load_and_aggregate_adapt(
    data_input_path: str,
    debug_limit=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[nx.Graph], pd.DataFrame]:
    """
    Load raw ADAPT results, attach graph_num, aggregate per graph, and build
    unique graph list for model inference.
 
    Returns:
        adapt_df      : raw ADAPT DataFrame with 'graph_num' column
        adapt_agg     : per-graph aggregated ADAPT stats
        graphs_unique : list of unique nx.Graph instances
        meta_df       : DataFrame with ['graph_name', 'graph_num'] aligned
                        to graphs_unique
    """
    from src.adapt_utils import get_combined_res_df  # local import to keep utils portable
 
    adapt_df = get_combined_res_df(data_input_path, debug_limit=debug_limit)
    adapt_df["graph_num"] = adapt_df["graph_name"].apply(graph_name_to_num)
 
    print(f"Total ADAPT rows      : {len(adapt_df)}")
    print(f"Unique graphs         : {adapt_df['graph_num'].nunique()}")
    print(f"Runs per graph (mean) : {adapt_df.groupby('graph_num').size().mean():.2f}")
 
    adapt_agg = adapt_df.groupby("graph_num").agg(
        graph_name        = ("graph_name",   "first"),
        adapt_ar_mean     = ("approx_ratio", "mean"),
        adapt_time_mean    = ("took_time",    "mean"),
        adapt_ar_best     = ("approx_ratio", "max"),
        adapt_ar_std      = ("approx_ratio", "std"),
        adapt_layers_mean = ("n_layers",     "mean"),
        adapt_layers_best = ("n_layers",     "min"),
        adapt_n_runs      = ("run",          "count"),
    ).reset_index()
 
    adapt_agg["adapt_ar_std"] = adapt_agg["adapt_ar_std"].fillna(0)
 
    unique_adapt_df        = adapt_df.drop_duplicates(subset="graph_num").reset_index(drop=True)
    graphs_unique, meta_df = load_graphs_from_adapt(unique_adapt_df)
 
    print(f"\nAggregated ADAPT shape : {adapt_agg.shape}")
    print(f"Graphs fed to model    : {len(graphs_unique)}")
 
    return adapt_df, adapt_agg, graphs_unique, meta_df

def preprocess_qaoa_df(qaoa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw QAOA runs → per-graph aggregated results

    Output columns:
        graph_num
        qaoa_ar_mean
        qaoa_ar_best
        qaoa_time_mean
        qaoa_layers
    """
    df = qaoa_df.copy()

    # Extract graph_num from "graph_1" → 1
    df["graph_num"] = df["graph_name"].str.split("_").str[-1].astype(int)

    # Aggregate per graph
    agg_df = (
        df.groupby("graph_num").agg(
            qaoa_ar_mean  = ("approx_ratio", "mean"),
            qaoa_ar_best  = ("approx_ratio", "max"),
            qaoa_time_mean= ("took_time", "mean"),
            qaoa_layers   = ("n_layers", "first"),  # same for all runs
        )
        .reset_index()
    )

    return agg_df
 
def build_results_df(
    meta_df: pd.DataFrame,
    cfg: dict,
    ar: pd.Series,
    layers: pd.Series,
    error_rate: pd.Series,
) -> pd.DataFrame:
    """
    Assemble a per-graph results DataFrame from metric series and config metadata.
 
    Columns: graph_name, graph_num, model, arch, method,
             model_ar, model_layers, model_error_rate
    """
    return pd.DataFrame({
        "graph_name"       : meta_df["graph_name"].values,
        "graph_num"        : meta_df["graph_num"].values,
        "model"            : cfg["resolved_name"],
        "arch"             : cfg["arch"],
        "method"           : cfg["method"],
        "model_ar"         : ar.values,
        "model_layers"     : layers.values,
        "model_error_rate" : error_rate.values,
    })
 
def build_final_df(
    adapt_agg: pd.DataFrame,
    model_results_df: pd.DataFrame,
    qaoa_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge ADAPT + model + optional QAOA (no diff columns)
    """

    # ------------------------
    # BASE MERGE
    # ------------------------
    final_df = adapt_agg.merge(model_results_df, on="graph_num")

    if "graph_name_x" in final_df.columns:
        final_df = final_df.rename(columns={"graph_name_x": "graph_name"}).drop(
            columns=["graph_name_y"], errors="ignore"
        )

    # ------------------------
    # ADD QAOA
    # ------------------------
    if qaoa_df is not None:
        qaoa_agg = preprocess_qaoa_df(qaoa_df)
        final_df = final_df.merge(qaoa_agg, on="graph_num", how="left")

    return final_df.sort_values("graph_num").reset_index(drop=True)
 
def build_summary_df(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to one row per model (clean version)
    """

    agg_dict = {
        "arch":              ("arch",             "first"),
        "method":            ("method",           "first"),
        "adapt_ar_mean":     ("adapt_ar_mean",    "mean"),
        "adapt_ar_best":     ("adapt_ar_best",    "mean"),
        "adapt_layers":      ("adapt_layers_mean","mean"),
        "adapt_time_mean":   ("adapt_time_mean","mean"),
        "model_ar":          ("model_ar",         "mean"),
        "model_error_rate":  ("model_error_rate", "mean"),
        "model_layers":      ("model_layers",     "mean"),
    }

    # Add QAOA if exists
    if "qaoa_ar_mean" in final_df.columns:
        agg_dict.update({
            "qaoa_ar_mean":   ("qaoa_ar_mean", "mean"),
            "qaoa_ar_best":   ("qaoa_ar_best", "mean"),
            "qaoa_layers":    ("qaoa_layers", "mean"),
            "qaoa_time_mean": ("qaoa_time_mean", "mean"),
        })

    return final_df.groupby("model").agg(**agg_dict).reset_index()


def maxcut_bruteforce(G):
    """
    Returns:
        best_energy (negative)
        best_bitstring (int)
    """
    n = G.number_of_nodes()
    edges = [(u, v, d.get("weight", 1.0)) for u, v, d in G.edges(data=True)]

    best_cut = -1
    best_state = 0

    # iterate over all bitstrings
    for state in range(1 << n):
        cut = 0

        for u, v, w in edges:
            # XOR trick: check if bits differ
            if ((state >> u) ^ (state >> v)) & 1:
                cut += w

        if cut > best_cut:
            best_cut = cut
            best_state = state

    # IMPORTANT: convert to NEGATIVE energy (match your dataset)
    best_energy = -best_cut

    return best_energy, best_state