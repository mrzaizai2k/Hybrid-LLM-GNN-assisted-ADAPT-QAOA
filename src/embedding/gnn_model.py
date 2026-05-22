import sys
sys.path.append("")

import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GINConv, AttentionalAggregation
from src.utils import read_config

class GNNGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        embedding_dim=500,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.project = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = self.pool(x, batch)
        return self.project(graph_emb)


def nx_to_pyg(graph):
    G = graph.copy()

    # ensure all edges have weight
    for u, v in G.edges():
        if "weight" not in G[u][v]:
            G[u][v]["weight"] = 1.0

    data = from_networkx(G)

    data.x = torch.tensor(
        np.vstack([G.nodes[n]["x"] for n in G.nodes()]),
        dtype=torch.float,
    )

    # optional: store edge weight tensor
    if "weight" in data:
        data.edge_weight = data.weight.float()

    return data

class GNN:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)["gnn"]

        self.node_feature_dim = self.config["NODE_FEATURE_DIM"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GNNGraphEncoder(
            in_dim=self.node_feature_dim,
            hidden_dim=self.config["HIDDEN_DIM"],
            embedding_dim=self.config["EMBEDDING_DIM"],
            num_layers=self.config["NUM_LAYERS"],
        ).to(self.device)

        self.model_path = self.config["MODEL_PATH"]
        self.load_model()

        print(f"GNN model initialized on device: {self.device}")

    def load_model(self):
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    # --------------------------------------------------
    # 🔑 KEY ADDITION: ensure node["x"] exists
    # --------------------------------------------------
    def _ensure_node_features(self, G: nx.Graph):
        for n in G.nodes():
            if "x" not in G.nodes[n]:
                G.nodes[n]["x"] = np.random.randn(self.node_feature_dim)

    def get_embedding(self, graphs):
        embeddings = []

        with torch.no_grad():
            for G in tqdm(graphs, desc="GNN"):
                self._ensure_node_features(G)

                data = nx_to_pyg(G).to(self.device)

                batch = torch.zeros(
                    data.num_nodes, dtype=torch.long, device=self.device
                )

                emb = self.model(data.x, data.edge_index, batch)
                emb = emb.squeeze(0).cpu().numpy()

                # ✅ CLIP then ROUND
                emb = np.clip(emb, 0.01, 0.99)
                emb = np.round(emb, 2)

                embeddings.append(emb)

        embeddings = np.stack(embeddings)
        print("GNN shape:", embeddings.shape)
        return embeddings


if __name__ == "__main__":

    np.random.seed(42)

    graphs = []
    num_graphs = 3
    num_nodes = 8
    edge_prob = 0.4

    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, edge_prob)
        for u, v in G.edges():
            G[u][v]["weight"] = np.random.uniform(0.5, 2.0)
        graphs.append(G)

    model = GNN()
    embeddings = model.get_embedding(graphs)

    embeddings = np.round(embeddings, 2)

    print("Embedding shape:", embeddings.shape)
    for i, emb in enumerate(embeddings):
        print(f"\nGraph {i} embedding (first 10 dims):")
        print(emb[:10])