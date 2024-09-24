from typing_extensions import Self
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, Linear, GCNConv
from torch.nn import ModuleList, Module
from torch import mean, stack

class HGNN(Module):

    def __init__(self, hid, out, layers) -> Self: # type: ignore
        super().__init__()

        self.convs = ModuleList()
        for _ in range(layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("activity", "follows", "activity"): GATConv((-1, -1), hid, add_self_loops=False),
                        ("activity", "has", "resource_static"): GATConv((-1,-1), hid, add_self_loops=False),
                        ('resource_dynamic', 'resource_delta', 'resource_dynamic'): GATConv((-1, -1), hid, add_self_loops=False),
                        ('activity', 'has', 'resource_dynamic') : GATConv((-1,-1),hid, add_self_loops=False)
                    },
                    aggr="sum",
                )
            )
        self.linear_nodes = Linear(hid, hid)
        
        self.fc = Linear(hid, out)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key,x in x_dict.items()}
        
        nodes_features = [self.linear_nodes(x_dict[key]).relu() for key in x_dict.keys()]
        
        for i in range(len(nodes_features)):
            nodes_features[i] = mean(nodes_features[i], dim=0)
        
        nodes_features = mean(stack(nodes_features),dim=0)
        nodes_features = self.fc(nodes_features)
        return nodes_features # {key : self.linear(x_dict[key]) for key in x_dict.keys()}, nodes_features