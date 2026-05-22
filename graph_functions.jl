import Graphs
import SimpleWeightedGraphs
import Random


function graph_to_adj_m(g)
    adj_matrix = Matrix(SimpleWeightedGraphs.adjacency_matrix(g))
    return adj_matrix
end


function graph_to_edgelist(g)
    weighted_edge_list = [
        (
            SimpleWeightedGraphs.src(e),
            SimpleWeightedGraphs.dst(e),
            SimpleWeightedGraphs.weight(e)
        ) for e in SimpleWeightedGraphs.edges(g)
    ]

    return weighted_edge_list
end 


function edgelist_to_graph(edgelist; num_vertices=0)

    if num_vertices == 0
        num_vertices = maximum(max(src, dst) for (src, dst, w) in edgelist)
    end
    g = SimpleWeightedGraphs.SimpleWeightedGraph(num_vertices)

    # Add edges with weights to the graph
    for (src, dst, w) in edgelist
        Graphs.add_edge!(g, src, dst, w)
    end
    
    return g
end

function generate_random_graph(
    n::Int;
    methods::Vector{String} = [
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
        "random_regular",
        "bipartite"
    ]
)
    """
    Generate a connected random graph using specified methods with random parameters.
    Ensures the graph is connected.

    Args:
        n (Int): Number of vertices in the graph (e.g., 10-12).
        methods (Vector{String}): List of graph generation methods to choose from.

    Returns:
        G (SimpleWeightedGraph): A randomly generated, connected graph.
        method (String): The name of the method used to generate the graph.
    """
    method = rand(methods)
    G = SimpleWeightedGraphs.SimpleWeightedGraph(n)

    while !Graphs.is_connected(G) || Graphs.ne(G) == 0
        if method == "erdos_renyi"
            p = rand(0.3:0.1:0.9)
            G = Graphs.erdos_renyi(n, p)

        elseif method == "barabasi_albert"
            m = rand(1:n - 1)
            G = Graphs.barabasi_albert(n, m)

        elseif method == "watts_strogatz"
            k = rand(2:n - 1)
            p = rand(0.1:0.1:1.0)
            G = Graphs.watts_strogatz(n, k, p)

        elseif method == "random_regular"
            d = rand(2:n - 1)
            G = Graphs.random_regular_graph(n, d)
        
        elseif method == "bipartite"
            n1 = rand(2:n - 1)
            n2 = n - n1
            G = Graphs.complete_bipartite_graph(n1, n2)
        end
    end
    
    G = SimpleWeightedGraphs.SimpleWeightedGraph(G)
    return G, method
end


function add_rand_weights_to_graph(
        g::SimpleWeightedGraphs.SimpleWeightedGraph;
        neg_weights::Bool,
    )
    n_nodes = Graphs.nv(g)
    g_weighted = SimpleWeightedGraphs.SimpleWeightedGraph(n_nodes)
    for e in SimpleWeightedGraphs.edges(g)
        s = SimpleWeightedGraphs.src(e)
        d = SimpleWeightedGraphs.dst(e)
        w = round(rand(), digits=2)
        while w == 0.0
            w = round(rand(), digits=2)
        end 
        if neg_weights
            w_sign = rand([-1,1])
            w = w * w_sign
        end
        SimpleWeightedGraphs.add_edge!(g_weighted, s, d, w)
    end
    return g_weighted
end


function norm_elist_weights(e_list)
    # Initialize a new list to store the scaled edges
    scaled_weighted_edge_list = Vector{Tuple{Int64, Int64, Float64}}()

    total_weight = 0
    
    # Iterate through each edge in the edge list
    for (node1, node2, weight) in e_list
        total_weight += abs(weight)
    end
    
    # Iterate through each edge in the edge list
    for (node1, node2, weight) in e_list
        # Scale the weight by the coefficient
        scaled_weight = weight / total_weight
        
        # Append the scaled edge to the new list
        push!(scaled_weighted_edge_list, (node1, node2, scaled_weight))
    end

    return scaled_weighted_edge_list, total_weight
end