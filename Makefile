train:
	cd nanoGPT && \
	python train_pad_gemb_ar_eval.py --train_config_path data/9_nodes_gnn/train_adapt_gpt_config.py --model gpt && \
	python train_pad_gemb_ar_eval.py --train_config_path data/9_nodes_gnn/train_adapt_gpt_config.py --model llama && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model llama && \
	python train_pad_gemb_ar_eval.py --train_config_path data/11_nodes_gnn/train_adapt_gpt_config.py --model llama && \
	python train_pad_gemb_ar_eval.py --train_config_path data/11_nodes_gnn/train_adapt_gpt_config.py --model gpt

11:
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/11_nodes --save_dir nanoGPT/data/11_nodes_feather --n_nodes 11 --embedding_method feather && \
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/11_nodes --save_dir nanoGPT/data/11_nodes_netlsd --n_nodes 11 --embedding_method netlsd && \
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/11_nodes --save_dir nanoGPT/data/11_nodes_gnn --n_nodes 11 --embedding_method gnn

9:
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/9_nodes --save_dir nanoGPT/data/9_nodes_feather --n_nodes 9 --embedding_method feather && \
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/9_nodes --save_dir nanoGPT/data/9_nodes_netlsd --n_nodes 9 --embedding_method netlsd && \
	python prepare_circ.py --adapt_results_dir ADAPT.jl_results/9_nodes --save_dir nanoGPT/data/9_nodes_gnn --n_nodes 9 --embedding_method gnn

# 	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model gpt && \
