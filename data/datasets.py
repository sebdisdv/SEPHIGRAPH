from typing_extensions import Self
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset, DataLoader


class Het_graph_data(Dataset):
    def __init__(self, prefix_graphs, labels) -> Self:
        self.X = prefix_graphs
        self.Y = labels

    # get the number of rows in the dataset
    def __len__(self):
        return len(self.Y)

    # get a row at a particular index in the dataset
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        Y = [item[1] for item in batch]
        return [data, Y]
