from models import models
from torch_geometric.data import HeteroData

def main():
    
    model = models.HGNN(64, 10, 2)
    


if __name__ == "__main__":
    main()