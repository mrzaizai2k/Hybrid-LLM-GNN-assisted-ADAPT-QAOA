import sys
sys.path.append("")

import time
from contextlib import nullcontext
import torch
from nanoGPT.model_pad_gemb import GPTConfig as GPTConfig_gemb, GPT as GPT_gemb
from nanoGPT.model_llama import LlamaConfig, Llama

import pandas as pd
from pathlib import Path

from src.circuit_util import (
    generate_circ_from_df,
    prepare_model_input,
    eval_adapt_gpt_circ_jl,
)

dtype_str_to_torch_dict = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
}


MODEL_MAP = {
    "gpt": (GPTConfig_gemb, GPT_gemb),
    "llama": (LlamaConfig, Llama),
}


class QAOA_GPT():

    def __init__(
        self,
        model_ckpt,
        data_dir,
        temp_folder="temp_data",
    ):

        self.data_dir = Path(data_dir)
        self.model_ckpt = Path(model_ckpt)
        self.temp_folder = Path(temp_folder)

        name = self.model_ckpt.name
        first = name.split("_")[0]

        # ---------- load config ----------
        if first in MODEL_MAP:
            self.model_type = first

            ckpt = torch.load(self.model_ckpt, map_location="cpu")
            cfg = ckpt["config"]

            self.pool_type = cfg.get("pool_type")
            self.n_nodes = cfg.get("n_nodes")
            self.embedding_method = cfg.get("embedding_method", "feather")
            self.seed = cfg.get("seed", 1337)

        else:
            self.model_type = "gpt" # default to gpt if not specified in filename and not in config file
            config_fpath = self.data_dir / "train_adapt_gpt_config.py"

            config_vars = {}
            with open(config_fpath) as f:
                exec(f.read(), config_vars)

            self.pool_type = config_vars["pool_type"]
            self.n_nodes = config_vars.get("n_nodes")
            self.embedding_method = "feather"
            self.seed = config_vars.get("seed", 1337)

        print(f"\nModel type: {self.model_type}")
        print(f"Pool type: {self.pool_type}")
        print(f"Embedding method: {self.embedding_method}")
        print(f"Number of nodes: {self.n_nodes}")

        # ---------- device ----------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dtype = (
            "bfloat16"
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else "float16"
        )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ptdtype = dtype_str_to_torch_dict[self.dtype]

        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        )

        self.meta = pd.read_pickle(self.data_dir / "meta.pkl")

        # ---------- model ----------
        self.model_config_class, self.model_class = MODEL_MAP[self.model_type]

        self.model = self.open_model(self.model_ckpt)

    # --------------------------------------------------

    def open_model(self, model_fpath):

        checkpoint = torch.load(model_fpath, map_location=self.device)

        conf = self.model_config_class(**checkpoint["model_args"])
        model = self.model_class(conf)

        state_dict = checkpoint["model"]

        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[10:]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval().to(self.device)

        return model

    def generate_circ_from_nx(
        self,
        graphs_container,
        calculate_classic_maxcut=True,
        n_samples_per_batch=50,
        num_samples=5,
        max_new_tokens=150,
        temperature=0.1,
        top_k=200,
        allow_larger_graphs=False,
    ):

        start_time = time.time()
        graphs_nx_df, graph_par_emb, emb_graph_id_to_idx_dict = prepare_model_input(
            graphs_container,
            calculate_classic_maxcut=calculate_classic_maxcut,
            embedding_method=self.embedding_method,
        )

        if self.device == "cpu":
            emb_dtype = "float"
        else:
            emb_dtype = self.dtype

        # print("graph_par_emb:", graph_par_emb)
        # print("len(graph_par_emb[0]):", len(graph_par_emb[0]))

        gc_df = generate_circ_from_df(
            graphs_nx_df,
            graph_emb_np=graph_par_emb,
            emb_graph_id_to_idx_dict=emb_graph_id_to_idx_dict,
            meta=self.meta,
            model=self.model,
            device=self.device,
            ctx=self.ctx,
            n_samples_per_batch=n_samples_per_batch,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            token_seq_col="token_seq_round_d2",
            normalize_weights_flag=False,
            emb_dtype=dtype_str_to_torch_dict[emb_dtype],
            allow_larger_graphs=allow_larger_graphs,
        )
        # print("gc_df:", gc_df.head())
        took_time = time.time() - start_time  # <-- add

        gc_df["took_time"] = took_time
        
        self.gc_df = gc_df
        self.graph_par_emb= graph_par_emb
   

        return gc_df

    def eval_circ_df_jl(
        self,
        qaoa_gpt_circ_df,
        adapt_gpt_path=".",
    ):

        qaoa_gpt_circ_eval_df = eval_adapt_gpt_circ_jl(
            qaoa_gpt_circ_df,
            n_nodes=self.n_nodes,
            adapt_gpt_path=adapt_gpt_path,
            temp_folder=self.temp_folder,
            pool_type=self.pool_type,
        )

        self.qaoa_gpt_circ_eval_df = qaoa_gpt_circ_eval_df
        output_columns_list = [
            "graph_prefix",
            "graph",
            "n_edges",
            "q_circuits",
            "adapt_gpt_energies",
        ]

        if "energy_mqlib" in qaoa_gpt_circ_df.columns:
            output_columns_list.append("energy_mqlib")

        #This is  enrgy calculated by gurobi (classical brute-force), which is the optimal solution for maxcut
        if "energy_gurobi" in qaoa_gpt_circ_df.columns: 
            output_columns_list.append("energy_gurobi")

        return qaoa_gpt_circ_eval_df[output_columns_list]