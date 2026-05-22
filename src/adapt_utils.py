import subprocess
import os
from datetime import datetime
from pathlib import Path
import json
from itertools import islice 
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from typing import Tuple

def split_list(lst, n):
    it = iter(lst)
    return [
        list(islice(it, i)) for i in [
            len(lst) // n + (1 if x < len(lst) % n else 0) for x in range(n)
        ]
    ]

def run_adapt_jl_parallel(
    script_dir,
    output_dir,
    input_graphs=None,
    n_workers=20,
    graphs_number=500,
    n_nodes=10,
    trials_per_graph=1,
    energy_tol_frac="0.03",
    max_params=50,
    weighted=True,
    normalize_weights=False,
    gamma_0="gamma0_grid_012325.json",
    pool_name="qaoa_mixer",
    use_floor_stopper=True,
    temp_folder='adapt_temp_data'
):
    # Environment setup
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"

    # Prepare log directory
    cur_date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_dir = os.path.join(output_dir, "logs", cur_date)
    os.makedirs(log_dir, exist_ok=True)

    cur_temp_dir = Path(temp_folder) / cur_date
    cur_temp_dir.mkdir(parents=True, exist_ok=True)
    
    input_graphs_list = []
    if input_graphs is None:
        input_graphs_list = ["Rand"] * n_workers 
    else:
        # Prepare input for multi-threading
        print(f"Splitting input graphs into {n_workers} parts")
        cur_input_graphs_dict = json.load(open(input_graphs))
        cur_input_graphs_dict_list = [
            dict(el) for el in split_list(cur_input_graphs_dict.items(), n_workers)
        ]
        for idx, graph_dict in enumerate(cur_input_graphs_dict_list):
            cur_temp_fname = cur_temp_dir / f"chunk_{idx}.json"
            with open(cur_temp_fname, 'w') as f:
                json.dump(graph_dict, f)
            input_graphs_list.append(cur_temp_fname)
    
    processes = []
    log_paths_list = []
    for i in range(n_workers):
        log_path = os.path.join(log_dir, f"worker_{os.uname().nodename}_{i}.log")
        log_paths_list.append(Path(log_path))
        print(f"Starting worker {i} on node: {os.uname().nodename}")
        
        # Construct the shell command as a single string
        cmd = f"""
        julia --compiled-modules=yes --project={script_dir}/ADAPT.jl {script_dir}/adapt_maxcut_run_1thread.jl \
            --output-dir {output_dir} \
            --graphs-number {graphs_number} \
            --n-nodes {n_nodes} \
            --trials-per-graph {trials_per_graph} \
            --energy-tol-frac {energy_tol_frac} \
            --max-params {max_params} \
            --weighted {str(weighted).lower()} \
            --run-qaoa true \
            --run-vqe false \
            --calc-h-eigen true \
            --run-diag-qaoa true \
            --normalize-weights {str(normalize_weights).lower()} \
            --g0 {gamma_0} \
            --op-pool {pool_name} \
            --use-floor-stopper {str(use_floor_stopper).lower()} \
            --graphs-input-json {input_graphs_list[i]} \
            2>&1 | tee {log_path}
        """

        # Start process with shell to allow piping through tee
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            env=env,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )
        processes.append(proc)

    # Wait for all workers to complete
    # for proc in processes:
    #     proc.wait()
    
    return log_paths_list, processes


def show_adapt_logs(log_paths, n_lines=10, pbar_only=False):
    for log_path in log_paths:
        print(f'Log: {log_path.stem}')
        with open(log_path, "r") as f:
            lines = f.readlines()
        if pbar_only:
            last_pbar_update = ''
            for line in lines:
                if 's/it' in line:
                    last_pbar_update = line
            print(last_pbar_update)
        else:
            print("".join(lines[-n_lines:]))
            print('-'*50)
        
    return None


def get_combined_res_df(
    results_fpath_list,
    debug_limit=None,
    n_workers=1
):
    ## ADAPT results
    if type(results_fpath_list) != list:
        results_fpath_list = [results_fpath_list]
        
    results_folders_list = [
        Path(e) for e in results_fpath_list
    ]
    print("Reading ADAPT.jl results from:")
    for res_fpath in results_folders_list:
        print('\t', str(res_fpath.absolute()))

    df_list = []
    df_list_all = []

    def open_df_from_res_csv(fname):
        #df_list = []
        try:
            cur_df = pd.read_csv(fname)
            cur_df['worker_id'] = fname.stem
            #df_list.append(cur_df)
        except Exception as e:
            print(f'{e} (file: {fname})')
            cur_df = None
        return cur_df

    for cur_dataset_res_path in results_folders_list:
        cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('res').glob('*.csv'))
        if debug_limit:
            cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
        df_list = Parallel(n_jobs=n_workers)(
            delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening ADAPT results ({cur_dataset_res_path.stem})')
        )
        df_list_all += df_list
    df_list  = [df for df in df_list_all if df is not None]
    print("df_list len:", len(df_list))

    full_run_df = pd.concat(df_list)
    full_run_df['prefix'] = full_run_df['worker_id'].apply(
        lambda x: x[:-15]
    )

    ## Graphs
    df_list = []
    df_list_all = []
    for cur_dataset_res_path in results_folders_list:
        cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('graphs').glob('*.csv'))
        if debug_limit:
            cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
        df_list = Parallel(n_jobs=n_workers)(
            delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening graphs ({cur_dataset_res_path.stem})')
        )
        df_list_all += df_list

        for df in df_list:
            if df is not None:
                if 'g_method' not in df.columns:
                    #print("Graphs were generated with older version of ADAPT-GPT preprocessor. Most likely, they are ER.")
                    df['g_method'] = "erdos_renyi"

    df_list  = [df for df in df_list_all if df is not None]
    print("df_list len:", len(df_list))


    full_run_graphs_df = pd.concat(df_list)
    full_run_graphs_df['edgelist_list'] = (
        full_run_graphs_df['edgelist_json'].apply(
            lambda x: json.loads(x)
        )
    )
    full_run_graphs_df['edgelist_list_len'] = (
        full_run_graphs_df['edgelist_list'].apply(
            lambda x: len(x)
        )
    )
    full_run_graphs_df['num_connected_comp'] = full_run_graphs_df['edgelist_list'].apply(
        lambda x: len(
            list(
                nx.connected_components(
                    nx.Graph([edge[:2] for edge in x])
                )
            )
        )
    )
    full_run_graphs_df['prefix'] = full_run_graphs_df['worker_id'].apply(
        lambda x: x[:-12]
    )

    print("Graphs count:")
    print(full_run_graphs_df['g_method'].value_counts())

    # Aggregating results
    print("Aggregating results...")

    full_run_df['approx_ratio'] = (
        full_run_df['energy'] / full_run_df['energy_mqlib']
    )

    full_run_agg_stat_df = (
        full_run_df.groupby(
            [
                #"graph_name",
                "graph_num", "run", "prefix", "method",
                'optimizer',
                "gamma0", "pooltype", "graph_name", 'n_nodes']
        )
        .agg(
            {
                #'pooltype': 'count',
                'energy': list,
                #'method': 'last',
                'took_time': 'last',
                'energy_mqlib': 'last',
                'generator_index_in_pool': list,
                'approx_ratio': 'last',
                #'approx_ratio_eig': 'last',
                'edge_weight_norm_coef': 'last',
                'β_coeff': list,
                'γ_coeff': list,
                'coeff': list,
                'energy_mqlib': 'last',
                'energy_eigen': 'last',
                #'n_nodes': 'last',
                'cut_mqlib': 'last',
                'cut_adapt': 'last',
                'cut_eig': 'last',
                'state_vect_adapt': 'last',
                'success_flag': 'last'
            }
        )
        .reset_index()
        .rename(
            columns={
                'pooltype': 'pooltype',
                'energy': 'energy_list',
                'generator_index_in_pool': 'op_list',
            }
        )
    )

    combined_res_df = pd.merge(
        left=full_run_agg_stat_df,
        right=full_run_graphs_df,
        left_on=['prefix', 'graph_num'],
        right_on=['prefix', 'graph_num'],
    )
    combined_res_df["n_layers"] = combined_res_df['energy_list'].apply(len)

    combined_res_df['graph_id'] = (
          combined_res_df['prefix']
        + '_^_'
        + combined_res_df['graph_num'].astype(str)
    )
    
    return combined_res_df

def compute_metrics_per_graph(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute evaluation metrics per graph instance.

    Each row in `df` corresponds to a graph and contains lists of:
        - q_circuits
        - adapt_gpt_energies
        - energy_gurobi (scalar)

    Returns:
        Tuple of three pd.Series (aligned with df.index):
            - ar: Mean approximation ratio per graph (NaN if no valid samples)
            - layers: Mean number of QAOA layers per graph
            - error_rate: Fraction of invalid samples (energy == 999) per graph
    """
    full_index = df.index

    # --- Layers ---
    df_expl = df.explode(["adapt_gpt_energies", "q_circuits"])
    layers = (
        df_expl.groupby(level=0)["q_circuits"]
        .apply(lambda xs: xs.apply(lambda x: x.count("new_layer_p")).mean())
        .reindex(full_index)
    )

    # --- Energy & error rate ---
    df_energy = df[["adapt_gpt_energies", "energy_gurobi"]].explode("adapt_gpt_energies")

    error_rate = (
        df_energy.groupby(level=0)["adapt_gpt_energies"]
        .apply(lambda x: (x == 999).sum() / len(x))
        .reindex(full_index, fill_value=0.0)
    )

    # --- Approximation ratio (AR) ---
    df_valid = df_energy[df_energy["adapt_gpt_energies"] != 999].copy()
    df_valid["ar"] = df_valid["adapt_gpt_energies"] / df_valid["energy_gurobi"]

    ar = (
        df_valid.groupby(level=0)["ar"]
        .mean()
        .reindex(full_index, fill_value=np.nan)
    )

    return ar, layers, error_rate

def compute_metrics(
    df: pd.DataFrame,
) -> Tuple[float, float, float]:
    """
    Compute dataset-level metrics by aggregating per-graph metrics.

    This function ensures each graph contributes equally (not biased by
    number of samples per graph).

    Returns:
        Tuple[float, float, float]:
            - avg_ar: Mean approximation ratio across graphs (ignores NaNs)
            - avg_error_rate: Mean error rate across graphs
            - avg_layers: Mean number of layers across graphs
    """
    ar, layers, error_rate = compute_metrics_per_graph(df)

    avg_ar = float(ar.mean(skipna=True))
    avg_error_rate = float(error_rate.mean())
    avg_layers = float(layers.mean())

    return avg_ar, avg_error_rate, avg_layers