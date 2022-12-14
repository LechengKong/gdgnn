from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from gnnfree.utils.graph import dgl_graph_to_gt_graph


class DatasetWithCollate(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def get_collate_fn(self):
        pass


class SingleGraphDataset(DatasetWithCollate):
    def __init__(self, graph):
        super().__init__()
        self.num_nodes = graph.num_nodes()
        self.graph = graph
        self.adj_mat = self.graph.adjacency_matrix(
            transpose=False, scipy_fmt="csr"
        )
        self.gt_g = dgl_graph_to_gt_graph(self.graph)
