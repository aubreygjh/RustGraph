import os
import random
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


def generateDataset(dataset, raw_file, device, snap_size, train_per, anomaly_per, x_dim):
    print('Generating data with anomaly for Dataset: ', dataset)
    edges = preprocessDataset(dataset, raw_file)
    edges = edges[:, 0:2].astype(dtype=int)
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

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
    for epoch in range(1, 50):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    x = n2v()
    # file_name_test = "./ano_generation/test_" + dataset + str(anomaly_per) + ".npy"
    # file_name_train = "./ano_generation/train_" + dataset + str(anomaly_per) + ".npy"
    # if os.path.exists(file_name_test) == False and os.path.exists(file_name_train)==False:
    #     synthetic_test, train = anomaly_generation2(train_per, anomaly_per, edges, n, m, seed=1)
    #     np.save(file_name_test, synthetic_test)
    #     np.save(file_name_train, train)
    # else:
    #     synthetic_test = np.load(file_name_test)
    #     train = np.load(file_name_train)
    # synthetic_test, train = anomaly_generation2(train_per, anomaly_per, edges, n, m, seed=1)
    synthetic_test, train, anomaly_num = edge_division(train_per, anomaly_per, edges, n, m)
    train_size = int(len(train) / snap_size + 0.5)
    test_size = int((len(synthetic_test)+anomaly_num) / snap_size + 0.5)
    print(train_size, test_size)
    # 计算每个时间步的异常数量
    anomaly_num_list = get_anomaly_num(anomaly_num, test_size)
    # 
    data_list = []
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = (ii+1) * snap_size
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
        if ii == 0:
            start_loc = 0
        else:
            start_loc = end_loc
        end_loc = start_loc + snap_size - anomaly_num_list[ii]
        edge_index = synthetic_test[start_loc:end_loc, 0:2]
        node_index = np.unique(edge_index)
        # 异常注入
        ano_num_1 = random.randint(0, anomaly_num_list[ii])
        ano_num_2 = anomaly_num_list[ii] - ano_num_1
        ori_adj_1 = {i:set() for i in node_index}
        ori_adj_2 = {i:set() for i in node_index}
        # 当前已经存在的顶点
        for edge in edge_index:
            ori_adj_1[edge[0]].add(edge[1])
            ori_adj_1[edge[1]].add(edge[0])
        for j in node_index:
            for jj in ori_adj_1[j]:
                ori_adj_2[j].union(ori_adj_1[jj])
                
        if ano_num_1 != 0:
            idx_1 = np.expand_dims(np.transpose(np.random.choice(node_index, m)) , axis=1)
            idx_2 = np.expand_dims(np.transpose(np.random.choice(node_index, m)) , axis=1)
            fake_edges_1 = np.concatenate((idx_1, idx_2), axis=1)
            fake_edges_1 = processEdges2(fake_edges_1, ori_adj_2, ano_num_1, edges, mode=0)
            fake_edges_1 = fake_edges_1[0:ano_num_1,:]
        else:
            fake_edges_1 = np.zeros([0,2])
        # 当前不存在的顶点
        if ano_num_2 != 0:
            new_node = np.array(list(set(vertices)-set(node_index)))
            idx_1 = np.expand_dims(np.transpose(np.random.choice(new_node, m)) , axis=1)
            idx_2 = np.expand_dims(np.transpose(np.random.choice(vertices, m)) , axis=1)
            fake_edges_2 = np.concatenate((idx_1, idx_2), axis=1)
            fake_edges_2 = processEdges2(fake_edges_2, ori_adj_2, ano_num_2, edges, mode=1)
            fake_edges_2 = fake_edges_2[0:ano_num_2,:]
        else:
            fake_edges_2 = np.zeros([0,2])
        # 综合两阶段的异常边
        fake_edges = np.concatenate((fake_edges_1, fake_edges_2), axis=0)
        fake_edges = np.concatenate((fake_edges, np.ones([np.size(fake_edges, 0),1])), axis=1)
        edge_index = np.concatenate((edge_index, fake_edges[:,0:2]), axis=0)
        y = np.concatenate((synthetic_test[start_loc:end_loc, 2], fake_edges[:,2]), axis=0)
        
        node_index = np.unique(edge_index)
        edge_attr = torch.randn(len(edge_index), 64)

        edge_index = torch.LongTensor(edge_index).t().contiguous()
        node_index = torch.LongTensor(node_index)
        y = torch.LongTensor(y)
        data = Data(x=x, edge_index=edge_index,node_index=node_index,y=y,edge_attr=edge_attr)
        data_list.append(data)
    
    return data_list, train_size

def get_anomaly_num(total_num, size):
    ano_per = []
    anomaly_num = []
    for i in range(size):
        ano_per.append(random.randint(10,20))
    total_per = sum(ano_per)
    for i in range(size):
        num = int(ano_per[i] * total_num / total_per + 0.5)
        anomaly_num.append(num)
    return anomaly_num 

def edge_division(ini_graph_percent, anomaly_percent, data, n, m):
    train_num = int(np.floor(ini_graph_percent * m))
    train = data[0:train_num, : ]
    test = data[train_num:, : ]
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    
    train = np.concatenate((train, np.zeros([np.size(train, 0),1])), axis=1)
    test = np.concatenate((test, np.zeros([np.size(test, 0),1])), axis=1)
    return train, test, anomaly_num

def processEdges2(fake_edges, data, total_num, all_data, mode):
    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)
    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]
    fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)
    fake_edges = fake_edges[idx_remove_dups]
    
    a = fake_edges.tolist()
    b = data
    b_2 = all_data.tolist()
    c = []
    
    if mode==1:
        for i in a:
            if i not in b_2 :
                c.append(i)
                if len(c) >= total_num:
                    break
    else:
        for i in a:
            if i[1] not in b[i[0]] and i not in b_2:
                c.append(i)
                if len(c) >= total_num:
                    break
    fake_edges = np.array(c)
    return fake_edges


def anomaly_generation2(ini_graph_percent, anomaly_percent, data, n, m,seed = 1):
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

    #data to adjacency_matrix
    #adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    # kk = 3 #3#10#42#42
    # sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    # labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # 真就直接随机生成
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    #fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # 移除掉self-loop以及真实边
    fake_edges = processEdges(fake_edges, data)

    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    # 按比例圈定要的异常边
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    # 按照总边数（测试正常+异常）圈定标签
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    # 随机选择异常边的位置
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    # 选定的位置定为1
    idx_test[anomaly_pos] = 1

    # 汇总数据，按照起点，终点，label的形式填充，并且把对应的idx找出
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    # coo:efficient for matrix construction ;  csr: efficient for arithmetic operations
    # coo+to_csr is faster for small matrix, but nearly the same for large matrix (size: over 100M)
    #train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] , train[:, 1])),shape=(n, n))
    # train_mat = coo_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])), shape=(n, n)).tocsr()
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n)
    # train_mat = train_mat + train_mat.transpose()
    train = np.concatenate((train, np.zeros([np.size(train, 0),1])), axis=1)
    print(f'Anomaly injection takes {time.time()-t0}s.')
    return synthetic_test, train #train_mat

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