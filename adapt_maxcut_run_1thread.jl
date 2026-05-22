import Graphs
import SimpleWeightedGraphs
import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis
import LinearAlgebra: norm, eigen, Diagonal
import SciPyOptimizers
import CSV
import DataFrames
using JuMP, MQLib
using Dates
import JSON
import Random
using ProgressBars
using ArgParse
using Multibreak

include("graph_functions.jl")

# Function to parse the command-line arguments
function parse_commandline()
    # Create a new argument parser
    s = ArgParseSettings()

    # Define named arguments
    @add_arg_table s begin
        "--output-dir"
        help = "Output directory"
        arg_type = String
        
        "--graphs-number"
        help = "Number of graphs"
        arg_type = Int
        default = 10
        
        "--trials-per-graph"
        help = "Number of times to run ADAPT per one graph"
        arg_type = Int
        default = 3
        
        "--n-nodes"
        help = "Number of nodes for generated graphs"
        arg_type = Int
        default = 8
        
        "--weighted"
        help = "Use weighted graphs. If false, all weights are set to 1."
        arg_type = Bool
        default = true
        
        "--use-negative-weights"
        help = "Use negative weights. Even when --weighted=false, weights can still be +1 or -1."
        arg_type = Bool
        default = false
        
        "--run-vqe"
        help = "Run ADAPT VQE"
        arg_type = Bool
        default = false
        
        "--run-qaoa"
        help = "Run ADAPT QAOA"
        arg_type = Bool
        default = true

        "--run-diag-qaoa"
        help = "Run DiagonalQAOA"
        arg_type = Bool
        default = false
        
        "--degen"
        help = "Run Degenerate ADAPT"
        arg_type = Bool
        default = true
        
        "--g0"
        help = "Gamma 0 parameter"
        arg_type = String
        default = "inv"
        
        "--max-params"
        help = "Cut-off value for number of parameters in output circuit"
        arg_type = Int
        default = 30
        
        "--energy-tol-frac"
        help = "Energy tolerance fraction for FloorStopper (if used)"
        arg_type = Float64
        default = 0.001
        
        "--scaling-coef"
        help = "Apply scaling to every weight in a graph to speedup convergence"
        arg_type = Float64
        default = 1.0
        
        "--normalize-weights"
        help = "Apply edge weight normalization to speedup convergence"
        arg_type = Bool
        default = false
        
        "--graphs-input-json"
        help = "Do not generate random graphs, use the provided json file instead to read graphs from"
        arg_type = String
        default = "N/A"
        
        "--calc-h-eigen"
        help = "Calculate eigenvalue decomposition (exact)"
        arg_type = Bool
        default = true
        
        "--save-state-vect"
        help = "Save the whole state vector"
        arg_type = Bool
        default = false
        
        "--optimizer"
        help = "Used optimizer. Possible options: BFGS, COBYLA"
        arg_type = String
        default = "BFGS"

        "--op-pool"
        help = "Operator pool to use. Options: qaoa_double_pool, qaoa_mixer. Default: qaoa_double_pool "
        arg_type = String
        default = "qaoa_double_pool"
        
        "--use-floor-stopper"
        help = "Use floor stopper callback"
        arg_type = Bool
        default = true
        
    end

    return parse_args(s)
end

args = parse_commandline()

#####
# Identifiers for the current worker
pid = getpid()
hostname = get(ENV, "HOSTNAME", "unknown")
#println("hostname: ", hostname, "; pid: ", pid)
current_datetime = now()
ts_string = string(Dates.format(current_datetime, "yy-mm-dd__HH_MM"))
#####

output_dir = args["output-dir"]
graphs_number = args["graphs-number"]
n_nodes = args["n-nodes"]
trials_per_graph = args["trials-per-graph"]
use_negative_weights = args["use-negative-weights"]
run_vqe = args["run-vqe"]
run_qaoa = args["run-qaoa"]
g0 = args["g0"]
max_params = args["max-params"]
energy_tol_frac = args["energy-tol-frac"]
weighted = args["weighted"]
diag_qaoa = args["run-diag-qaoa"]
degen = args["degen"]
json_graphs_fname = args["graphs-input-json"]
calc_h_eigen = args["calc-h-eigen"]
scaling_coef = args["scaling-coef"]
norm_weights = args["normalize-weights"]
save_state_vect = args["save-state-vect"]
optimizer = args["optimizer"]
op_pool = args["op-pool"]
use_floor_stopper = args["use-floor-stopper"]

println("Running ADAPT with parameters (worker: $hostname, pid: $pid):")
for (arg,val) in args
    println("  $arg  =>  $val")
end

struct ModalSampleTracer <: ADAPT.AbstractCallback end

function (tracer::ModalSampleTracer)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList,
    ::ADAPT.Observable, ψ0::ADAPT.QuantumState,
)
    ψ = ADAPT.evolve_state(ansatz, ψ0)      # THE FINAL STATEVECTOR
    imode = argmax(abs2.(ψ))                # MOST LIKELY INDEX
    zmode = imode-1                         # MOST LIKELY BITSTRING (as int)

    push!( get!(trace, :modalsample, Any[]), zmode )
    return false
end

function exact_ground_state_energy(H, n)
    Emin = Ref(Inf); ketmin = Ref(KetBitString{n}(0))
    for v in 0:1<<n-1
        ket = KetBitString{n}(v)
        vec = SparseKetBasis{n,ComplexF64}(ket => 1)
        Ev = real((H*vec)[ket])
        if Ev < Emin[]
            Emin[] = Ev
            ketmin[] = ket
        end
    end
    ψ0 = Vector(SparseKetBasis{n,ComplexF64}(ketmin[] => 1))
    E0 = Emin[]
    
    ρ = abs2.(ψ0)                    # THE FINAL PROBABILITY DISTRIBUTION
    pmax, imax = findmax(ρ)
    ketmax = KetBitString(n, imax-1) # THE MOST LIKELY BITSTRING
    
    return E0, string(ketmax)
end 

function maxcut_matrix_from_adjacency(A::Matrix{Float64})
    # Step 1: Compute the Degree matrix D
    D = Diagonal(sum(A, dims=2)[:])
    
    # Step 2: Compute the Laplacian matrix L
    L = D - A
    
    # Step 3: Compute the Max-Cut matrix Q
    Q = L ./ 2
    
    return Q
end

function get_mqlib_energy_from_adj_m(adj_matrix)
    # MQLib block
    mqlib_model = Model(MQLib.Optimizer)
    MQLib.set_heuristic(mqlib_model, "BURER2002")
    Q = maxcut_matrix_from_adjacency(adj_matrix)
    n_nodes = size(adj_matrix)[1]
    @variable(mqlib_model, x[1:n_nodes], Bin)
    @objective(mqlib_model, Max, x' * Q * x)
    JuMP.optimize!(mqlib_model)
    maxcut_val_mqlib = JuMP.objective_value(mqlib_model)
    
    #exact_energy_val = -0.5 * maxcut_val_mqlib
    
    exact_energy_val = -2 * maxcut_val_mqlib
    
    solution = JuMP.value.(x)

    return solution, exact_energy_val
end

function scale_elist_weights(e_list, coef)
    # Initialize a new list to store the scaled edges
    scaled_weighted_edge_list = Vector{Tuple{Int64, Int64, Float64}}()

    # Iterate through each edge in the edge list
    for (node1, node2, weight) in e_list
        # Scale the weight by the coefficient
        scaled_weight = weight * coef
        
        # Append the scaled edge to the new list
        push!(scaled_weighted_edge_list, (node1, node2, scaled_weight))
    end

    return scaled_weighted_edge_list
end

if !isdir(""*output_dir*"/hams")
    mkpath(""*output_dir*"/hams")
end
if !isdir(""*output_dir*"/res")
    mkpath(""*output_dir*"/res")
end
if !isdir(""*output_dir*"/graphs")
    mkpath(""*output_dir*"/graphs")
end
if !isdir(""*output_dir*"/traces")
    mkpath(""*output_dir*"/traces")
end

start_time = time()

# RUN MANY ADAPT-QAOA TRIALS, CHOOSING RANDOMLY WHEN THE GRADIENTS ARE DEGENERATE
results_df = DataFrames.DataFrame()
hams_df = DataFrames.DataFrame()
graphs_df = DataFrames.DataFrame(
    graph_num = Int[],
    g_method = String[],
    edgelist_json = String[],
    H_frob_norm = Float64[],
)
traces_df = DataFrames.DataFrame()

# main loop

if json_graphs_fname != "N/A"
    println("Loading graphs from: $json_graphs_fname")
    json_graphs_dict = JSON.Parser.parsefile(json_graphs_fname);
    graphs_number = length(json_graphs_dict)
    graph_names_list = collect(keys(json_graphs_dict))
end

iter = ProgressBar(1:graphs_number)
set_description(iter, "Graphs on: "*hostname*"; pid: "*string(pid)*":")
#for graph_num in 1:graphs_number
@multibreak begin
    for graph_num in iter
        norm_coef = 1.0

        if json_graphs_fname == "N/A"
            cur_graph_name = "Graph_$graph_num"
#             prob = Random.rand(prob_list)
#             g = rand_weigh_graph_generator(n_nodes, prob, weighted=weighted, neg_weights=use_negative_weights)

#             while Graphs.ne(g) == 0
#                 println("Generated empty graph! Trying again")
#                 g = rand_weigh_graph_generator(n_nodes, prob, weighted=weighted, neg_weights=use_negative_weights)
#             end
            
            g_unweighted, g_method = generate_random_graph(
                n_nodes,
                methods=[
                    "erdos_renyi",
                    #"barabasi_albert",
                    #"watts_strogatz",
                    #"random_regular",
                    #"bipartite"
                ]
            )
            
            if weighted
                g = add_rand_weights_to_graph(
                    g_unweighted,
                    neg_weights=use_negative_weights
                )
            else
                g = g_unweighted
            end
        else
            prob = "N/A"
            cur_graph_name = graph_names_list[graph_num]
            cur_graph_elist = json_graphs_dict[cur_graph_name]["elist"]
            global n_nodes = json_graphs_dict[cur_graph_name]["n_nodes"]
            g = edgelist_to_graph(cur_graph_elist, num_vertices=n_nodes)
            g_method = "input_file"
        end

        # BUILD OUT THE POOL
        if op_pool == "qaoa_double_pool"
            pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_nodes); pooltype = "qaoa_double_pool"
        elseif op_pool == "qaoa_mixer"
            pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_mixer(n_nodes); pooltype = "qaoa_mixer"
        elseif op_pool == "two_local_pool"
            pool = ADAPT.Pools.two_local_pool(n_nodes); pooltype = "two_local_pool"
        else
            throw(error("op_pool is not valid."))
        end

        #pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_ext_double_pool(n_nodes); pooltype = "qaoa_ext_double_pool_v2"
        pool_size = length(pool)
        println("Pool size: $pool_size")

        e_list = graph_to_edgelist(g)
        #println(typeof(e_list))

        edgelist_json = JSON.json(e_list)

        # scaling down the weights

        if scaling_coef != 1.0
            println("Scaling all edgelist weights with $scaling_coef")
            e_list = scale_elist_weights(e_list, scaling_coef)
            #println(e_list)
            #println(typeof(e_list))
        end

        if norm_weights
            e_list, norm_coef = norm_elist_weights(e_list)
            println("Normalizing all edgelist weights by $norm_coef")
            #println(e_list)
            #println(typeof(e_list))
        end

        println("\nGraph name: $cur_graph_name;\nNumber of edges: $(Graphs.ne(g));\nNumber of nodes: $(Graphs.nv(g));\nGenerator: $g_method.\n")

        e_exact_eig = -999.0
        bitstring_exact_eig = "N/A"
        
        # BUILD OUT THE PROBLEM HAMILTONIAN

        H_spv = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list)
        h_frob_norm = norm(Matrix(H_spv))
        if calc_h_eigen
            e_exact_eig, bitstring_exact_eig = exact_ground_state_energy(H_spv, n_nodes)
            if scaling_coef != 1.0
                e_exact_eig = e_exact_eig / scaling_coef
            end

            if norm_coef != 1.0
                e_exact_eig = e_exact_eig * norm_coef
            end
        end

        if diag_qaoa
            H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv)
        else
            H = H_spv
        end

        push!(graphs_df, (graph_num, g_method, edgelist_json, h_frob_norm))

        ##########
        # MQLib block

        exact_cut_solution, exact_energy_val = (
            get_mqlib_energy_from_adj_m(
                graph_to_adj_m(g)
            )
        )


        exact_cut_solution_string = join(
            map(
                x -> x == 1.0 ? "1" : "0",
                exact_cut_solution
            )
        )


        # if scaling_coef != 1.0
        #     exact_energy_val = exact_energy_val / scaling_coef
        # end

        energy_tol = abs(energy_tol_frac * exact_energy_val)
        scaled_exact_energy_val = exact_energy_val / norm_coef
        scaled_energy_tol = energy_tol / norm_coef

        # println("Generator data type: ", typeof(pool[1]))
        # println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")

        # SELECT THE PROTOCOLS

        if degen
            adapt = ADAPT.Degenerate_ADAPT.DEG_ADAPT
        else
            adapt = ADAPT.VANILLA
        end

        # SELECT THE CALLBACKS
        callbacks = [
            ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores, :elapsed_time, :g_norm),
            ADAPT.Callbacks.ParameterTracer(),
            #ADAPT.Callbacks.Printer(:energy, :selected_index, :selected_score),
            ModalSampleTracer(),
            #op_pool != "qaoa_mixer" ? ADAPT.Callbacks.ScoreStopper(1e-3) : nothing,
            #ADAPT.Callbacks.ScoreStopper(1e-3),
            ADAPT.Callbacks.ParameterStopper(max_params),
            # ADAPT.Callbacks.SlowStopper(1.0, 3),
            # ADAPT.Callbacks.TimeStopper(soft_time_limit),
        ]
        
        # Conditionally add ScoreStopper
        if op_pool != "qaoa_mixer"
            push!(callbacks, ADAPT.Callbacks.ScoreStopper(1e-5))
        end
        
        if use_floor_stopper
            push!(callbacks, ADAPT.Callbacks.FloorStopper(scaled_energy_tol, scaled_exact_energy_val))
        end
        
        println("Number of callbacks: $(length(callbacks))")

        #optimizers_list = ["BFGS", "COBYLA"]
        #optimizers_list = ["COBYLA"]
        optimizers_list = ["BFGS"]

        println("Exact energy (MQLib): $exact_energy_val")
        ##########

        for trial_num = 1:trials_per_graph

            for opt_name in optimizers_list
                if opt_name == "BFGS"
                    vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-4)
                elseif opt_name == "COBYLA"
                    vqe = SciPyOptimizers.SciPyOptimizer("COBYLA";
                        tol=1e-6,       # Not exactly sure what this entails, honestly...
                        rhobeg=1,     # Reasonable initial changes to values.
                        maxiter=10000,   # Maximum number of iterations per optimization.
                        disp=false,      # Use scipy's own verbose printing.
                    )
                end

                #opt_name = "COBYLA_2pi/10";

                println("Running with optimizer: $opt_name")

                ### VQE BLOCK ###

                if run_vqe

                    cur_start_time = time()

                    # INITIALIZE THE REFERENCE STATE
                    ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0)

                    # gamma0=0.5/Graphs.ne(g)

                    # println("Using gamma0 = 0.5/Graphs.ne(g) = $gamma0")

                    # INITIALIZE THE ANSATZ AND TRACE
                    #ansatz = ADAPT.Basics.Ansatz(gamma0, H) 
                    ansatz = ADAPT.Basics.Ansatz(1.0, pool) 
                    #= the first argument is a hyperparameter and can in principle 
                    be set to values other than 0.1 =#
                    trace = ADAPT.Trace()

                    # RUN THE ALGORITHM
                    success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
                    println(success ? "Success!" : "Failure - optimization didn't converge.")

                    cur_end_time = time()
                    cur_elapsed_time = cur_end_time - cur_start_time

                    # # RESULTS
                    # if !success
                    #     continue
                    # end

                    bitstrings_list = []
                    for z in trace[:modalsample][2:end]
                        cur_bitstring = bitstring(z)[end-n_nodes:end]
                        #println(cur_bitstring)
                        push!(bitstrings_list, string(cur_bitstring))
                    end

                    energies_list = trace[:energy][trace[:adaptation][2:end]]
                    if scaling_coef != 1.0
                        energies_scaled_list = energies_list ./ scaling_coef
                    else
                        energies_scaled_list = energies_list
                    end

                    if norm_coef != 1.0
                        energies_scaled_list = energies_list * norm_coef
                    else
                        energies_scaled_list = energies_list
                    end

                    println("VQE ketmax:", bitstrings_list[end])
                    println("eig ketmax:", bitstring_exact_eig)

                    # SAMPLE MOST LIKELY BITSTRING
                    ψ = ADAPT.evolve_state(ansatz, ψ0)      # THE FINAL STATEVECTOR
                    ρ = abs2.(ψ)                            # THE FINAL PROBABILITY DISTRIBUTION
                    pmax, imax = findmax(ρ)
                    ketmax = KetBitString(n_nodes, imax-1)        # THE MOST LIKELY BITSTRING

                    state_vect_json = JSON.json(round.(ρ, digits=4))

                    # SAVE THE TRACE
                    try
                       #throw(error("hello"))
                       cur_res_df = DataFrames.DataFrame(
                            :method => "vqe",
                            :graph_name => cur_graph_name,
                            :graph_num => graph_num,
                            :run => trial_num,
                            :n_nodes => n_nodes,
                            :gamma0 => -999.0,
                            :optimizer => opt_name,
                            :pooltype => pooltype,
                            :edge_weight_scaling_coef => scaling_coef,
                            :edge_weight_norm_coef => norm_coef,
                            :generator_index_in_pool => trace[:selected_index][1:end-1], 
                            :β_coeff => -999.0,
                            :γ_coeff => -999.0,
                            :coeff => ansatz.parameters,
                            :energy => energies_scaled_list,
                            #:energy_bfgs => trace[:energy], # cast to json string 
                            :energy_mqlib => exact_energy_val,
                            :energy_eigen => e_exact_eig,
                            :cut_mqlib => "$exact_cut_solution_string",
                            :cut_eig => bitstring_exact_eig,
                            :cut_adapt => bitstrings_list,
                            :state_vect_adapt => state_vect_json,
                            #:took_time => sum(trace[:elapsed_time]),
                            #:took_time => sum(trace[:elapsed_time][trace[:adaptation][2:end]]),
                            :took_time => cur_elapsed_time,
                            :success_flag => success,
                        )
                        append!(results_df, cur_res_df)
                    catch err
                       @error "ERROR: " exception=(err, catch_backtrace())
                    end
                    cur_trace_df = DataFrames.DataFrame(
                        :method => "vqe",
                        :graph_name => cur_graph_name,
                        :graph_num => graph_num,
                        :run => trial_num,
                        :trace_json => JSON.json(trace),
                    )
                    #println(JSON.json(trace))
                    append!(traces_df, cur_trace_df)
                end

                ### QAOA BLOCK ###

                if run_qaoa

                    # INITIALIZE THE REFERENCE STATE
                    ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0)

                    if g0 == "inv"
                        gamma0_list = [1/Graphs.ne(g)]
                    elseif occursin(".json", g0)
                        println("For QAOA reading gamma_0 values from: $g0")
                        gamma0_list = JSON.Parser.parsefile(g0);
                        @assert gamma0_list isa AbstractVector "gamma_0 json file does not contain a vector!"
                        gamma0_list = Array{Float64}(gamma0_list)
                    else
                        gamma0_list = [parse(Float64, g0)]
                    end

                    for gamma0 in gamma0_list

                        cur_start_time = time()

                        println("For QAOA using gamma_0 = $gamma0, trial: $trial_num/$trials_per_graph, graph N: $graph_num/$graphs_number")

                        if diag_qaoa
                            ansatz = ADAPT.ADAPT_QAOA.DiagonalQAOAAnsatz(gamma0, pool, H)
                        else
                            ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(gamma0, H)
                        end
                        #= the first argument is a hyperparameter and can in principle 
                        be set to values other than 0.1 =#
                        trace = ADAPT.Trace()

                        # RUN THE ALGORITHM
                        #println("RUNNING ADAPT QAOA!!!")
                        success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)

                        println(success ? "Success!" : "Failure - optimization didn't converge.")

                        cur_end_time = time()
                        cur_elapsed_time = cur_end_time - cur_start_time

                        #println(keys(trace))

                        # # RESULTS
                        # if !success
                        #     continue
                        # end

                        # DISPLAY MOST LIKELY BITSTRINGS FROM EACH ADAPTATION
                        #println("Most likely bitstrings after each adaption:")

                        bitstrings_list = []
                        for z in trace[:modalsample][2:end]
                            cur_bitstring = bitstring(z)[end-n_nodes+1:end]
                            #println(cur_bitstring)
                            push!(bitstrings_list, cur_bitstring)
                        end

                        energies_list = trace[:energy][trace[:adaptation][2:end]]
                        if scaling_coef != 1.0
                            energies_scaled_list = energies_list ./ scaling_coef
                        else
                            energies_scaled_list = energies_list
                        end

                        if norm_coef != 1.0
                            energies_scaled_list = energies_list * norm_coef
                        else
                            energies_scaled_list = energies_list
                        end

                        # println("QAOA ketmax:", bitstrings_list[end])
                        # println("eig ketmax:", bitstring_exact_eig)

                        # SAMPLE MOST LIKELY BITSTRING
                        ψ = ADAPT.evolve_state(ansatz, ψ0)      # THE FINAL STATEVECTOR
                        ρ = abs2.(ψ)                            # THE FINAL PROBABILITY DISTRIBUTION
                        pmax, imax = findmax(ρ)
                        ketmax = KetBitString(n_nodes, imax-1)        # THE MOST LIKELY BITSTRING

                        state_vect_json = JSON.json(round.(ρ, digits=4))

                        # if optimization ends prematurely, sometimes we have length mismatch
                        beta_coefs_list = ansatz.β_parameters
                        gamma_coefs_list = ansatz.γ_parameters
                        generator_index_in_pool_list = trace[:selected_index][1:end-1]

                        if length(beta_coefs_list) != length(generator_index_in_pool_list)
                            pop!(beta_coefs_list)
                        end

                        if length(gamma_coefs_list) != length(generator_index_in_pool_list)
                            pop!(gamma_coefs_list)
                        end

                        n_layers = length(generator_index_in_pool_list)
                        el_time = round(cur_elapsed_time, digits=2)
                        approx_ratio = energies_scaled_list[end] / exact_energy_val

                        println("final energy:\t$(energies_scaled_list[end]) (through trace)")
                        println("Took time: $el_time sec.;\nN layers: $n_layers;\ng0 = $gamma0;\nar = $approx_ratio")

                        # SAVE THE TRACE
                        try
                           #throw(error("hello"))
                           cur_res_df = DataFrames.DataFrame(
                                :method => "qaoa",
                                :graph_name => cur_graph_name,
                                :graph_num => graph_num,
                                :run => trial_num,
                                :n_nodes => n_nodes,
                                :gamma0 => gamma0,
                                :optimizer => opt_name,
                                :pooltype => pooltype,
                                :edge_weight_scaling_coef => scaling_coef,
                                :edge_weight_norm_coef => norm_coef,
                                :generator_index_in_pool => generator_index_in_pool_list, 
                                :β_coeff => beta_coefs_list,
                                :γ_coeff => gamma_coefs_list,
                                :coeff => -999.0,
                                :energy => energies_scaled_list,
                                :energy_mqlib => exact_energy_val,
                                :energy_eigen => e_exact_eig,
                                :cut_mqlib => "$exact_cut_solution_string",
                                :cut_eig => bitstring_exact_eig,
                                :cut_adapt => bitstrings_list,
                                :state_vect_adapt => state_vect_json,
                                #:took_time => sum(trace[:elapsed_time]),
                                #:took_time => sum(trace[:elapsed_time][trace[:adaptation][2:end]]),
                                :took_time => cur_elapsed_time,
                                :success_flag => success,
                            )
                            append!(results_df, cur_res_df)
                        catch err
                           @error "ERROR: " exception=(err, catch_backtrace())
                        end
                        cur_trace_df = DataFrames.DataFrame(
                            :method => "qaoa",
                            :optimizer => opt_name,
                            :graph_name => cur_graph_name,
                            :graph_num => graph_num,
                            :run => trial_num,
                            :trace_json => JSON.json(trace),
                        )

                        #println(JSON.json(trace))
                        append!(traces_df, cur_trace_df)

                        #------------------#
                        # Early stopping condition
                        if (approx_ratio >= 1 - energy_tol_frac)
                            println("Good approx ratio achieved with g0 = $gamma0, skipping other g0s")
                            break; break; break
                            # g0;  opt;   trial; 
                        end
                        #------------------#

                    end
                end
            end
        end

        if (json_graphs_fname == "N/A") && (!diag_qaoa)
            # DOES NOT WORK IF N_NODES VARIES ACROSS GRAPHS
            H_df = DataFrames.DataFrame(H)
            H_df[!, :graph_num] .= graph_num
            append!(hams_df, H_df)
            # WRITE THE HAMILTONIAN TO A FILE
            ham_file = ""*output_dir*"/hams/worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_Ham.csv"
            CSV.write(ham_file, hams_df)
        end

        # WRITE THE ADAPT-QAOA RESULTS TO A FILE
        results_file = ""*output_dir*"/res/worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_adapt_qaoa_res.csv"
        CSV.write(results_file, results_df)

        # WRITE GRAPHS TO A FILE
        graphs_file = ""*output_dir*"/graphs/worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_graphs_json.csv"
        CSV.write(graphs_file, graphs_df)

        # WRITE TRACES TO A FILE
        traces_file = ""*output_dir*"/traces/worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_traces_json.csv"
        CSV.write(traces_file, traces_df)

    end
end

end_time = time()
elapsed_time = end_time - start_time
println("\nhostname: ", hostname, "; pid: ", pid, "; The script took: ", elapsed_time, " seconds.\n")
