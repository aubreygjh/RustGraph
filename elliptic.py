import os
import torch
import pandas as pd
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip


class Elliptic(InMemoryDataset):
    r"""
    This dataset is the network of Bitcoin transactions from the 
    `"Anti-Money Laundering in Bitcoin: Experimenting with Graph
    Convolutional Networks for Financial Forensics"
    <https://arxiv.org/abs/1102.2166>`_ paper.
    Each node represents a transaction, and edges represent the flow 
    of Bitcoin between two transactions. Around 23% of the nodes in 
    the dataset have been labeled as being created by a “licit” or 
    “illicit” entity. Missing node labels are coded -1. Node features 
    comprise local and aggregated information about the transactions.
    
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://uofi.box.com/shared/static/vhmlkw9b24sxsfwh5in9jypmx2azgaac.zip'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']
        # return [
        #     os.path.join('elliptic_bitcoin_dataset', file) for file in
        #     ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']
        # ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def num_classes(self):
        return 2

    def download(self):
        file = download_url(self.url, self.raw_dir)
        extract_zip(file, self.raw_dir)
        os.unlink(file)

    def process(self):
        file_features = os.path.join(self.raw_dir,  'elliptic_txs_features.csv')
        # file_features = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_features.csv')
        df = pd.read_csv(file_features, index_col=0, header=None)
        x = torch.from_numpy(df.to_numpy()).float()

        file_classes = os.path.join(self.raw_dir, 'elliptic_txs_classes.csv')
        # file_classes = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_classes.csv')
        df = pd.read_csv(file_classes, index_col='txId', na_values='unknown').fillna(0) - 1
        y = torch.from_numpy(df.to_numpy()).view(-1).long()
        num_nodes = y.size(0)

        df_idx = df.reset_index().reset_index().drop(columns='class').set_index('txId')
        file_edges = os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        # file_edges = os.path.join(self.raw_dir, 'elliptic_bitcoin_dataset', 'elliptic_txs_edgelist.csv')

        df = pd.read_csv(file_edges).join(df_idx, on='txId1', how='inner')
        df = df.join(df_idx, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])
        edge_index = torch.from_numpy(df.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # torch.save((data,slices), self.processed_paths[0])
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return 'Elliptic()'