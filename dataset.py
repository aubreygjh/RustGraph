import os
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_tar, extract_gz, extract_zip
from data import *

class DynamicGraphAnomaly(InMemoryDataset):
    url = {'email':'https://nrvis.com/download/data/dynamic/email-dnc.zip',
            'as_topology':'https://nrvis.com/download/data/dynamic/tech-as-topology.zip',
            'btc_alpha':'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',
            'btc_otc':'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz',
            'uci':'http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2',
            'digg':'http://konect.cc/files/download.tsv.munmun_digg_reply.tar.bz2',
            'hepth':['https://snap.stanford.edu/data/cit-HepTh.txt.gz', 'https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz'],
            'enron':''}
    def __init__(self, root, name, args, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = name
        self.name = name + "_" + str(args.anomaly_ratio) + "_" + str(args.train_ratio) + "_" + str(args.noise_ratio) + "_" + str(args.x_dim)
        self.snap_size = args.snap_size
        self.train_ratio = args.train_ratio
        self.anomaly_ratio = args.anomaly_ratio
        self.noise_ratio = args.noise_ratio
        self.device = args.device
        self.x_dim = args.x_dim
        super(DynamicGraphAnomaly, self).__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=self.device)
        self.train_size = torch.load(self.processed_paths[1])
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset, self.name)
    
    @property
    def raw_file_names(self):
        if self.dataset == 'email':
            return ['email-dnc.edges']
        elif self.dataset == 'as_topology':
            return ['tech-as-topology.edges']
        elif self.dataset == 'btc_alpha':
            return ['soc-sign-bitcoinalpha.csv']
        elif self.dataset == 'btc_otc':
            return ['soc-sign-bitcoinotc.csv']
        elif self.dataset == 'uci':
            return ['opsahl-ucsocial/out.opsahl-ucsocial']
        elif self.dataset =='digg':
            return ['munmun_digg_reply/out.munmun_digg_reply']
       

    @property
    def processed_file_names(self):
        return ['data.pt', 'train_size.pt']

    def download(self):
        file = download_url(self.url[self.dataset], self.raw_dir)
        if self.dataset in ['email', 'as_topology']:
            extract_zip(file, self.raw_dir)
        elif self.dataset in ['uci', 'digg']:
            extract_tar(file, self.raw_dir, mode='r:bz2')
        elif self.dataset in ['btc_alpha', 'btc_otc']:
            extract_gz(file, self.raw_dir)        
        os.unlink(file)
      

    def process(self):
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_list, train_size = generateDataset(self.dataset, raw_file, self.device, self.snap_size, self.train_ratio, self.anomaly_ratio, self.noise_ratio, self.x_dim)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(train_size, self.processed_paths[1])

    def __repr__(self):
        return self.name

