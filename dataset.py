import os
from numpy.core.fromnumeric import transpose
import torch
import numpy as np
import pandas as pd
from torch._C import dtype
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip, extract_tar
from preprocess import preprocess, from_networkx

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

class UCI(InMemoryDataset):
    url = 'http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2'

    def __init__(self, root, transform=None, pre_transform=None):
        super(UCI, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def raw_file_names(self):
        return ['opsahl-ucsocial/out.opsahl-ucsocial']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        file = download_url(self.url, self.raw_dir)
        extract_tar(file, self.raw_dir, mode='r:bz2')
        os.unlink(file)

    def process(self):
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        preprocess(raw_file)
        graphs = np.load(os.path.join(self.raw_dir, "graphs.npz"), allow_pickle=True)['graph']
        features = np.load(os.path.join(self.raw_dir, "features.npz"), allow_pickle=True)['feats']
 
        # adj_matrices = map(lambda x: nx.adjacency_matrix(x), graphs)

        data_list = []
        for i,_ in enumerate(graphs):
            data = from_networkx(graphs[i], features[i], group_edge_attrs=['date'])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def __repr__(self):
        return 'UCI()'

class Digg(InMemoryDataset):
    url = 'http://konect.cc/files/download.tsv.munmun_digg_reply.tar.bz2'

    def __init__(self, root, transform=None, pre_transform=None):
        super(Digg, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ['munmun_digg_reply/out.munmun_digg_reply']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        file = download_url(self.url, self.raw_dir)
        extract_tar(file, self.raw_dir, mode='r:bz2')
        os.unlink(file)

    def process(self):
        # # Read data into huge `Data` list.
        # file = os.path.join(self.raw_dir,  'munmun_digg_reply/out.munmun_digg_reply')
        # df = pd.read_csv(file,sep=' ',skiprows=[0],header=None)
        # df = df.sort_values(by=3,ascending=True)
        # start_time = df.min()[3]
        # end_time = df.max()[3]
        # time_span = end_time - start_time
        # interval = 50000
        # df = df.to_numpy(dtype=np.int32)
        
        # edge_indice = {}
        # cnt=[0]*1000
        # for it in df:
        #     graph_idx = int((it[3] - start_time) / interval)
        #     cnt[graph_idx]+=1
            
        #     if edge_indice.get(graph_idx) is None:
        #         edge_indice[graph_idx] = np.expand_dims(it[0:2], axis=0)
        #     else:
        #         edge_indice[graph_idx] = np.concatenate((edge_indice[graph_idx], np.expand_dims(it[0:2], axis=0)),axis=0)
        # print(cnt)
     

        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return 'Digg()'