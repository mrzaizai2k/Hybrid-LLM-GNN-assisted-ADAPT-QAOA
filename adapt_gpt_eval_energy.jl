ENV["OPENBLAS_NUM_THREADS"] = "1"
import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm
import Graphs
import JSON
import JuMP, MQLib
using ProgressBars
import SimpleWeightedGraphs

using Base.Threads

include("eval_ansatz.jl")

if length(ARGS) < 4
    println("Usage: julia adapt_gpt_eval_energy.jl <input_fpath> <output_fpath> <n_nodes> <pool_type>")
    exit(1)
end

input_fpath = ARGS[1]
output_fpath = ARGS[2]
n_nodes = parse(Int, ARGS[3])
pool_type = ARGS[4]

adapt_gpt_out_list = JSON.parsefile(
    input_fpath
);

iter = ProgressBar(1:length(adapt_gpt_out_list))

@threads for graph_idx in iter
    
    adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]
    edgelist = adapt_gpt_out_dict["graph_w_jl"];
    adapt_gpt_energies_list = []
    adapt_gpt_bitstrings_list = []
    
    for i in 1:(length(adapt_gpt_out_dict["q_circuits"]) + 1)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            #println(i)
            generated_list = adapt_gpt_out_dict["q_circuits"][i]
        else
            #println("ADAPT")
            generated_list =  adapt_gpt_out_dict["adapt_circuit"]
        end

        E_final = 999
        best_bitstring = ""

        try
            E_final, best_bitstring = suppress_output(
                eval_ansatz,
                edgelist,
                generated_list,
                n_nodes,
                pool_type,
            )
        catch
        end
        #println(i)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            append!(adapt_gpt_energies_list, E_final);
            push!(adapt_gpt_bitstrings_list, best_bitstring)
        else
            #print("$i ADAPT")
            adapt_gpt_out_dict["ADAPT_energy_round"] = E_final;
            adapt_gpt_out_dict["ADAPT_bitstring_round"] = best_bitstring;
        end
    end
    adapt_gpt_out_dict["adapt_gpt_energies"] = adapt_gpt_energies_list;
    adapt_gpt_out_dict["adapt_gpt_bitstrings"] = adapt_gpt_bitstrings_list;
end

## Saving

adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);

open(output_fpath,"w") do f 
    write(f, adapt_gpt_out_list_json) 
end
