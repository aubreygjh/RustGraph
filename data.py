import os
import time
from datetime import datetime as dt
import datetime
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
def hepth(link_file, date_file, raw_file):
    node_date = dict()
    with open(date_file) as f:
        lines = f.read().splitlines()
        for l in lines:
            if l[0] == '#':
                continue
            node, date = l.split()
            node = int(node)
            date = dt.strptime(date,"%Y-%m-%d")
            node_date[node] = date.timestamp()
    links = []
    with open(link_file) as f:
        lines = f.read().splitlines()
        for l in lines:
            if l[0] == '#':
                continue
            x, y = map(int, l.split())
            
            if y in node_date:
                links.append((x,y,node_date[y]))
            
    links.sort(key =lambda x: x[2])
    links = np.array(links)
    np.savetxt(raw_file, links)
    return 

def preprocessDataset(dataset,raw_file):
    print('Preprocess dataset: ' + dataset)
    if dataset in ['digg', 'uci', 'as_topology']:
        edges = np.loadtxt(raw_file, dtype=float, comments='%', delimiter=' ')
        edges = edges[edges[:, 3].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)

    elif dataset in ['email']:
        with open(raw_file, encoding='utf-8-sig') as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 2].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)

    elif dataset in ['btc_alpha', 'btc_otc']:
        with open(raw_file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 3].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)
    
    elif dataset == 'hepth':
        with open(raw_file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(' ')] for row in lines]    
        edges = np.array(edges)
        edges = edges[edges[:, 2].argsort()]
        edges = edges[:, 0:2].astype(dtype=int)

    for ii in range(len(edges)):
        x0 = edges[ii][0]
        x1 = edges[ii][1]
        if x0 > x1:
            edges[ii][0] = x1
            edges[ii][1] = x0

    # edges = edges[np.nonzero([x[0] != x[1] for x in edges])].tolist() #remove self-loop
    # aa, idx = np.unique(edges, return_index=True, axis=0) #remove duplicate edges
    edges = np.array(edges)
    # edges = edges[np.sort(idx)]

    vertexs, edges = np.unique(edges, return_inverse=True)
    edges = np.reshape(edges, [-1, 2])
    print('vertex:', len(vertexs), ' edge: ', len(edges))
    print(edges.max(),edges.min())
    return edges

def n2v_train(edges, x_dim, device, n, epoch_num):
    edge_index = torch.LongTensor(edges).t().contiguous()
    n2v = Node2Vec(edge_index=edge_index, embedding_dim=x_dim,walk_length=25,context_size=25, num_nodes=n).to(device)
    loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(list(n2v.parameters()), lr=0.01)
    def train():
        n2v.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    for epoch in range(1, epoch_num):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    x = n2v()
    return x

def generateDataset(dataset, raw_file, device, snap_size, train_per, anomaly_per, x_dim):
    print('Generating data with anomaly for Dataset: ', dataset)
    edges = preprocessDataset(dataset, raw_file)
    edges = edges[:, 0:2].astype(dtype=int)
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)
    
    # edge_index = torch.LongTensor(edges).t().contiguous()
    # n2v = Node2Vec(edge_index=edge_index, embedding_dim=x_dim,walk_length=25,context_size=25, num_nodes=n).to(device)
    # loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
    # optimizer = torch.optim.Adam(list(n2v.parameters()), lr=0.01)
    # def train():
    #     n2v.train()
    #     total_loss = 0
    #     for pos_rw, neg_rw in loader:
    #         optimizer.zero_grad()
    #         loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     return total_loss / len(loader)
    # for epoch in range(1, 100):
    #     loss = train()
    #     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    # x = n2v()
    if dataset == 'digg':
        epoch_num = 75
    else:
        epoch_num = 50
    x = n2v_train(edges, x_dim, device, n, epoch_num)
    file_name_test = f"./ano_generation/test_{dataset}_{anomaly_per}_{train_per}_{snap_size}.npy"
    file_name_train = f"./ano_generation/train_{dataset}_{anomaly_per}_{train_per}_{snap_size}.npy"
    if os.path.exists(file_name_test) == False and os.path.exists(file_name_train)==False:
        synthetic_test, train = anomaly_generation2(train_per, anomaly_per, edges, n, m, seed=1)
        np.save(file_name_test, synthetic_test)
        np.save(file_name_train, train)
    else:
        synthetic_test = np.load(file_name_test)
        train = np.load(file_name_train)
    # synthetic_test, train = anomaly_generation2(train_per, anomaly_per, edges, n, m, seed=1)
    train_size = int(len(train) / snap_size + 0.5)
    test_size = int(len(synthetic_test) / snap_size + 0.5)
    print(train_size, test_size)

    data_list = []
    for ii in range(train_size):
        print('train:', ii)
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size
        edge_index = train[start_loc:end_loc, 0:2]
        node_index = np.unique(edge_index)
        y = train[start_loc:end_loc, 2]
        edge_attr = torch.randn(len(edge_index), 64)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        node_index = torch.LongTensor(node_index)
        y = torch.LongTensor(y)
        data = Data(x=x, edge_index=edge_index,node_index=node_index,y=y, edge_attr=edge_attr)
        data_list.append(data)
    for ii in range(test_size):
        print('test:', ii)
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size
        edge_index = synthetic_test[start_loc:end_loc, 0:2]
        node_index = np.unique(edge_index)
        y = synthetic_test[start_loc:end_loc, 2]
        edge_attr = torch.randn(len(edge_index), 64)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        node_index = torch.LongTensor(node_index)
        y = torch.LongTensor(y)
        data = Data(x=x, edge_index=edge_index,node_index=node_index,y=y,edge_attr=edge_attr)
        data_list.append(data)
     
    return data_list, train_size
    

def anomaly_generation2(ini_graph_percent, anomaly_percent, data, n, m,seed = 1):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train_mat: the training network with square matrix format, the training
                        network edges for initial model training;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # The actual generation method used for Netwalk(shown in matlab version)
    # Abort the SpectralClustering
    np.random.seed(seed)
    print('[%s] generating anomalous dataset...\n'% datetime.datetime.now())
    print('[%s] initial network edge percent: %.2f, anomaly percent: %.2f.\n'%(datetime.datetime.now(),
          ini_graph_percent , anomaly_percent ))
    t0 = time.time()
    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    synthetic_test = generate_anomaly(data, test, n, m, anomaly_percent)
    synthetic_train = generate_anomaly(data, train, n, m, 0.5)

    print(f'Anomaly injection takes {time.time()-t0}s.')
    return synthetic_test, synthetic_train 


def generate_anomaly(raw_data, inject_data, n, m, anomaly_ratio):
    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # 真就直接随机生成
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    #fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # 移除掉self-loop以及真实边
    fake_edges = processEdges(fake_edges, raw_data)

    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    # 按比例圈定要的异常边
    anomaly_num = int(np.floor(anomaly_ratio * np.size(inject_data, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    # 按照总边数（测试正常+异常）圈定标签
    labels = np.zeros([np.size(inject_data, 0) + anomaly_num, 1], dtype=np.float32)
    # randsample: sample without replacement
    # it's different from datasample!

    # 随机选择异常边的位置
    anomaly_pos = np.random.choice(np.size(labels, 0), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    # 选定的位置定为1
    labels[anomaly_pos] = 1

    # 汇总数据，按照起点，终点，label的形式填充，并且把对应的idx找出
    synthetic_data = np.concatenate((np.zeros([np.size(labels, 0), 2], dtype=np.float32), labels), axis=1)
    idx_anomalies = np.nonzero(labels.squeeze() == 1)
    idx_normal = np.nonzero(labels.squeeze() == 0)
    synthetic_data[idx_anomalies, 0:2] = anomalies
    synthetic_data[idx_normal, 0:2] = inject_data
    return synthetic_data

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # b:list->set
    # Time cost rate is proportional to the size

    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]

    fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    for i in a:
        if i not in b:
            c.append(i)
    fake_edges = np.array(c)
    return fake_edges