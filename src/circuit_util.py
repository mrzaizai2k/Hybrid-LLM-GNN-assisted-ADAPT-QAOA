import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import json
from typing import Tuple
import warnings
from gurobipy import Model, GRB
from src.get_embedding import get_embedding

def check_if_nx_graph_is_weighted(graph_nx):
    return all('weight' in graph_nx[u][v] for u, v in graph_nx.edges)

def extract_graph(token_seq):
    graph_seq = []

    for idx, tok in enumerate(token_seq):
        graph_seq.append(tok)
        if tok == 'end_of_graph':
            break
    adapt_seq = token_seq[idx+1:-1]
    return graph_seq, adapt_seq

def circ_sanity_check(cur_q_circ):
    
    lr_sep_list = cur_q_circ[0::4]
    op_idx_list = cur_q_circ[1::4]

    num_vals = cur_q_circ[2::4] + cur_q_circ[3::4]

    if any(
        [type(el) != int for el in op_idx_list]
    ):
        #print('wrong op_idx_list')
        return False

    if any(
        [type(el) != str for el in lr_sep_list]
    ):
        #print('wrong lr_sep_list')
        return False
    
    if len(cur_q_circ) % 4:
        #print('Wrong length')
        return False

    return True

def generate_circ_from_df(
    test_run_df,
    graph_emb_np,
    emb_graph_id_to_idx_dict,
    meta,
    model,
    device,
    ctx,
    n_samples_per_batch,
    num_samples = 5,
    max_new_tokens = 200,
    temperature = 0.1,
    top_k = 200,
    token_seq_col = 'token_seq_round_d2',
    normalize_weights_flag = False,
    emb_dtype=torch.bfloat16,
    allow_larger_graphs = False,  # NEW: set True to handle OOV edge tokens via modulo remap
):
    if graph_emb_np is not None and emb_graph_id_to_idx_dict is not None:
        gemb_flag = True
    else:
        gemb_flag = False
    
    stoi, itos = meta['stoi'], meta['itos']

    # --- encode/decode: two modes depending on allow_larger_graphs ---
    if not allow_larger_graphs:
        # Original strict encoding — will KeyError on unseen tokens
        encode = lambda s: [stoi[c] for c in s]
    else:
        # Modulo-remap OOV edge tokens so larger graphs can be encoded
        known_edge_tokens = [k for k in stoi.keys() if isinstance(k, tuple)]
        if not known_edge_tokens:
            raise ValueError("No edge tuple tokens found in stoi; cannot build remap table.")
        max_known_node = max(max(k) for k in known_edge_tokens)

        def _remap_token(c):
            if c in stoi:
                return stoi[c]
            if isinstance(c, tuple):
                remapped = tuple(sorted(n % (max_known_node + 1) for n in c))
                if remapped in stoi:
                    return stoi[remapped]
            warnings.warn(
                f"OOV token {c!r} could not be remapped (no match after modulo); using token 0.",
                stacklevel=2,
            )
            return 0

        encode = lambda s: [_remap_token(c) for c in s]

    decode = lambda l: [itos[i] for i in l]
    # --- end encode/decode ---

    n_edges_to_count_dict = test_run_df['edgelist_list_len'].value_counts().to_dict()
    
    adapt_gpt_out_list_dict = defaultdict(list)
    x_list_dict = defaultdict(list)
    graph_emb_dict = defaultdict(list)
    y_dict = dict()
    
    pbar = tqdm(n_edges_to_count_dict.items())
    
    for n_edges, n_graphs in pbar:
        pbar.set_description(f"Inference. Current batch: n_edges: {n_edges}, n_graphs: {n_graphs}")
        cur_test_run_df = test_run_df[test_run_df['edgelist_list_len'] == n_edges]
        
        for row_idx, graph_df_row in cur_test_run_df.iterrows():
            start, adapt_seq = extract_graph(graph_df_row[token_seq_col])
            start_ids = encode(start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            x_list_dict[n_edges].append(x)

            if gemb_flag:
                cur_graph_idx = emb_graph_id_to_idx_dict[graph_df_row['graph_id']]
                graph_emb_dict[n_edges].append(
                    torch.tensor(graph_emb_np[cur_graph_idx], dtype=emb_dtype, device=device)
                )
    
            adapt_gpt_out_dict = dict()
            adapt_gpt_out_dict['graph'] = start[1:-1]
            adapt_gpt_out_dict['n_edges'] = graph_df_row['edgelist_list_len']
            adapt_gpt_out_dict['q_circuits'] = []
            adapt_gpt_out_dict['adapt_circuit'] = adapt_seq
            adapt_gpt_out_dict['adapt_full_ar'] = graph_df_row['approx_ratio']
            adapt_gpt_out_dict['graph_prefix'] = graph_df_row['graph_id']
            if 'energy_mqlib' in graph_df_row:
                adapt_gpt_out_dict['energy_mqlib'] = graph_df_row['energy_mqlib']
            if 'energy_gurobi' in graph_df_row:
                adapt_gpt_out_dict['energy_gurobi'] = graph_df_row['energy_gurobi']
            adapt_gpt_out_dict['label'] = graph_df_row['label']
            adapt_gpt_out_list_dict[n_edges].append(adapt_gpt_out_dict)
        
        cur_batch_torch = torch.vstack(x_list_dict[n_edges])
        
        if gemb_flag:
            cur_emb_batch_torch = torch.vstack(graph_emb_dict[n_edges])
    
        total_samples = cur_batch_torch.size(0)
        n_batches = (total_samples + n_samples_per_batch - 1) // n_samples_per_batch
    
        y_list = []
        
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc='Internal batch progress', disable=True):
                start_idx = i * n_samples_per_batch
                end_idx = min((i + 1) * n_samples_per_batch, total_samples)
                
                mini_batch = cur_batch_torch[start_idx:end_idx]
                mini_batch_repeated = mini_batch.repeat(num_samples, 1)

                if gemb_flag:
                    mini_emb_batch = cur_emb_batch_torch[start_idx:end_idx]
                    mini_emb_batch_repeated = mini_emb_batch.repeat(num_samples, 1)
        
                with ctx:
                    if gemb_flag:
                        y = model.generate(
                            mini_batch_repeated,
                            mini_emb_batch_repeated,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k
                        )
                    else:
                        y = model.generate(
                            mini_batch_repeated,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k
                        )
        
                y_list.append(y.detach().cpu())
        
        y_dict[n_edges] = torch.cat(y_list, dim=0)

    ### trimming the records (removing garbage after EOS)
    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        cur_full_y_tensor = y_dict[n_edges]
        
        for graph_idx in range(len(cur_adapt_gpt_out_list)):
            cur_y_tensor = cur_full_y_tensor[graph_idx::len(cur_adapt_gpt_out_list)]
            
            for k in range(num_samples):
                cur_gen_result = decode(cur_y_tensor[k].tolist())
                cur_circ = []
                circ_flag = 0
                for idx, tok in enumerate(cur_gen_result):
                    if tok == 'end_of_graph':
                        circ_flag = 1
                    if circ_flag:
                        cur_circ.append(tok)
                    if tok == 'eos':
                        break
                cur_adapt_gpt_out_list[graph_idx]['q_circuits'].append(cur_circ[1:-1])

    ### flattening the circ list
    adapt_gpt_test_samples_list = []
    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        adapt_gpt_test_samples_list += cur_adapt_gpt_out_list

    if len(adapt_gpt_test_samples_list) == 0:
        print("Warning: No graphs to process. Returning empty DataFrame.")
        return pd.DataFrame()

    for idx in range(len(adapt_gpt_test_samples_list)):
        q_circ_filt_list = []
        for circ in adapt_gpt_test_samples_list[idx]['q_circuits']:
            circ_sanity_check(circ)
            q_circ_filt_list.append(circ)
        adapt_gpt_test_samples_list[idx]['q_circuits'] = q_circ_filt_list

    for gr_dict in adapt_gpt_test_samples_list:
        graph_jl_list = []
        graph_edges_list = gr_dict['graph'][::2]
        graph_weights_list = gr_dict['graph'][1::2]
    
        graph_w_norm = sum(graph_weights_list) if normalize_weights_flag else 1.0
        
        for edge_idx, edge in enumerate(graph_edges_list):
            cur_edge = list(edge)
            cur_edge += [graph_weights_list[edge_idx] / graph_w_norm]
            graph_jl_list.append(cur_edge)
    
        gr_dict['graph_w_jl'] = graph_jl_list
        gr_dict['graph_weight_norm'] = graph_w_norm

    adapt_gpt_test_samples_filt_list = [
        rec for rec in adapt_gpt_test_samples_list if rec  # pos_flag always 1; keep hook for future filters
    ]

    return pd.DataFrame(adapt_gpt_test_samples_filt_list)

def fix_new_layer_p(df):
    """
    Convert operator indices like 11.0 -> 11 when they appear
    immediately after 'new_layer_p' in any list or nested list column.
    """

    def fix_sequence(seq):
        if not isinstance(seq, list):
            return seq

        fixed = []
        i = 0
        while i < len(seq):
            if (
                seq[i] == "new_layer_p"
                and i + 1 < len(seq)
                and isinstance(seq[i + 1], (int, float))
            ):
                fixed.append(seq[i])

                op_val = seq[i + 1]

                # If float but actually integer-valued (11.0, 45.0, etc.)
                if isinstance(op_val, float) and op_val.is_integer():
                    fixed.append(int(op_val))
                else:
                    fixed.append(op_val)

                i += 2
            else:
                fixed.append(seq[i])
                i += 1

        return fixed

    def fix_value(val):
        if isinstance(val, list):
            # Nested list case (e.g. q_circuits)
            if len(val) > 0 and isinstance(val[0], list):
                return [fix_sequence(v) for v in val]
            return fix_sequence(val)
        return val

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(fix_value)

    return df

def eval_adapt_gpt_circ_jl(
    adapt_gpt_res_df,
    adapt_gpt_path,
    temp_folder,
    n_nodes,
    n_threads=4,
    pool_type="qaoa_double_pool",
):

    formatted_timestamp = datetime.now().strftime('%Y-%m-%d__%H_%M_%S')

    adapt_gpt_path = Path(adapt_gpt_path).resolve()
    temp_folder = Path(temp_folder).resolve()
    temp_folder.mkdir(parents=True, exist_ok=True)

    prefix = f'adapt_gpt_res_{formatted_timestamp}_df'
    in_fname_path = temp_folder / f"{prefix}.json"
    out_fname_path = temp_folder / f"{prefix}_jl.json"

    adapt_gpt_res_df = fix_new_layer_p(adapt_gpt_res_df) # should be removed
    adapt_gpt_res_df.to_json(in_fname_path, orient="records")

    adapt_jl_path = (adapt_gpt_path / "ADAPT.jl").resolve()
    script_path = (adapt_gpt_path / "adapt_gpt_eval_energy.jl").resolve()

    JULIA_BIN = "/opt/julia-1.12.1/bin/julia"

    cmd = [
        JULIA_BIN,
        "-t", str(n_threads),
        f"--project={adapt_jl_path}",
        str(script_path),
        str(in_fname_path),
        str(out_fname_path),
        str(n_nodes),
        pool_type,
    ]

    print("\n===== DEBUG INFO =====")
    print("CWD:", os.getcwd())
    print("Command:")
    print(" ".join(cmd))
    print("======================\n")

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    ret = process.wait()
    print("\nJulia return code:", ret)

    if not out_fname_path.exists():
        raise RuntimeError("Julia finished but output file was not created")

    return pd.read_json(out_fname_path)

def elist_to_nx(input_elist, jl_idx_shift=True):
    elist = []
    if jl_idx_shift:
        for u,v,w in input_elist:
            elist.append((u-1,v-1,w))
    else:
        elist = input_elist
    
    G = nx.Graph()
    G.add_weighted_edges_from(elist)
    
    return G

def nx_to_elist(nx_graph, jl_idx_shift=True):
    weighted = check_if_nx_graph_is_weighted(nx_graph)
    if not weighted:
        raise ValueError(
            "Current version of QAOA-GPT does not support unweighted graphs. "
            "Weights w are expected to be sampled from U(0,1)."
        )
    shifted_elist = []
    for edge_idx, (n1, n2) in enumerate(nx_graph.edges):
        cur_e_weight = nx_graph[n1][n2]['weight']
        if jl_idx_shift:
            # match Julia indexing 
            n1 = n1+1
            n2 = n2+1
        shifted_elist.append((n1, n2, cur_e_weight))
    graph_nx_from_edges = nx.from_edgelist(nx_graph.edges)
    n_nodes = graph_nx_from_edges.number_of_nodes()

    return {
        "elist": shifted_elist,
        "n_nodes": n_nodes
    }

def gurobi_max_cut_val_from_nx(graph_nx):

    model = Model("Max-Cut")
    model.setParam('OutputFlag', False) 
    model.setParam(GRB.Param.TimeLimit, 10)
    variables = {}
    for node in graph_nx.nodes:
        variables[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

    objective = 0
    for u,v,w in graph_nx.edges(data="weight"):
        objective -= w*((2*variables[v]*variables[u]) - (variables[v] + variables[u]))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    solution = [variables[node].x for node in graph_nx.nodes]
    
    return -model.ObjVal


def seq_tokenize_graph(elist):
    tok_list = ['bos']
    for n1, n2, w in elist:
        tok_list += [tuple(sorted([n1,n2])), w]
    tok_list.append('end_of_graph')
    return tok_list

def prepare_model_input(
    graphs_container,
    calculate_classic_maxcut=True,
    embedding_method='feather',
):
    
    if type(graphs_container) == list:
        graphs_edgelist_list_dict = {
            f'graph_{i}':g for i,g in enumerate(graphs_container)
        }
    elif type(graphs_container) == dict:
        graphs_edgelist_list_dict = graphs_container
    else:
        raise ValueError("Only list or dict containers are supported for input graphs!")
        
    graphs_nx_dict = defaultdict(dict)

    for name, nx_graph in tqdm(graphs_edgelist_list_dict.items(), desc='Preparing graphs...'):
        nx_elist_dict = nx_to_elist(nx_graph)
    
        graphs_nx_dict[name]['elist'] = nx_elist_dict['elist']
        graphs_nx_dict[name]['n_nodes'] = nx_elist_dict['n_nodes']
        if calculate_classic_maxcut:
            graphs_nx_dict[name]['energy_gurobi'] = gurobi_max_cut_val_from_nx(nx_graph)

    graphs_nx_df = pd.DataFrame(graphs_nx_dict).T.reset_index(names='graph_id')
    graphs_nx_df['token_seq_round_d2'] = graphs_nx_df['elist'].apply(seq_tokenize_graph)
    graphs_nx_df['edgelist_list_len'] = graphs_nx_df['elist'].apply(len)
    graphs_nx_df['approx_ratio'] = None
    graphs_nx_df['label'] = 'test_interactive'
    graphs_nx_df['edgelist_json'] = graphs_nx_df['elist'].apply(lambda x: json.dumps(x))

    print(f"Performing {embedding_method} embedding")
    graph_par_emb, emb_graph_idx_to_id_dict = get_embedding(
        graphs_nx_df,
        method=embedding_method,
    )
    
    emb_graph_id_to_idx_dict = {v:k for k,v in emb_graph_idx_to_id_dict.items()}
    
    graphs_nx_df['has_emb'] = graphs_nx_df['graph_id'].apply(
        lambda x: True if x in emb_graph_id_to_idx_dict else False
    )
    
    graphs_nx_df = graphs_nx_df[
        graphs_nx_df['has_emb']
    ]
    
    graphs_nx_df['graph_id'].apply(lambda x: x[:2]).value_counts()
    
    return graphs_nx_df, graph_par_emb, emb_graph_id_to_idx_dict


