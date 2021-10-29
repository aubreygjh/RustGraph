# import dill
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
# from scipy.sparse import csr_matrix
import time as tm
import torch
import torch_geometric

def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    all_nodes = list(set(all_nodes))
    print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1
    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])

        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
            #features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = np.squeeze(np.array(features_remap))
        # features_remap = csr_matrix(np.squeeze(np.array(features_remap))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap)


def preprocess(raw_file):
    links = []
    ts = []
    ctr = 0
    node_cnt = 0
    node_idx = {}
    idx_node = []

    with open(raw_file) as f:
        lines = f.read().splitlines()
        for l in lines:
            if l[0] == '%':
                continue
                
            x, y, e, t = map(int, l.split())
            # print (x,y,e,t)
            timestamp = datetime.fromtimestamp(t)
            ts.append(timestamp)
            
            ctr += 1
            if ctr % 100000 == 0:
                print (ctr)
                
            if x not in node_idx:
                node_idx[x] = node_cnt 
                node_cnt += 1
                
            if y not in node_idx:
                node_idx[y] = node_cnt 
                node_cnt += 1
        
            links.append((node_idx[x],node_idx[y], timestamp))

    print ("Min ts", min(ts), "max ts", max(ts))    
    print ("Total time span: {} days".format((max(ts) - min(ts)).days))
    links.sort(key =lambda x: x[2])

    SLICE_DAYS = 2
    START_DATE = min(ts) + timedelta(5)
    END_DATE = max(ts) - timedelta(60)

    slices_links = defaultdict(lambda : nx.MultiGraph())
    slices_features = defaultdict(lambda : {})

    print ("Start date", START_DATE)
    print ("End Date", END_DATE)
    slice_id = -1
    # Split the set of links in order by slices to create the graphs. 
    for (a, b, time) in links:
        prev_slice_id = slice_id
        
        datetime_object = time
        if datetime_object < START_DATE:
            continue
        if datetime_object > END_DATE:
            break
            days_diff = (END_DATE - START_DATE).days
        else:
            days_diff = (datetime_object - START_DATE).days
            
        
        slice_id = days_diff // SLICE_DAYS

        if slice_id == 1+prev_slice_id and slice_id > 0:
            slices_links[slice_id] = nx.MultiGraph()
            slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
            assert (len(slices_links[slice_id].edges()) ==0)
            #assert len(slices_links[slice_id].nodes()) >0

        if slice_id == 1+prev_slice_id and slice_id ==0:
            slices_links[slice_id] = nx.MultiGraph()

        if a not in slices_links[slice_id]:
            slices_links[slice_id].add_node(a)
        if b not in slices_links[slice_id]:
            slices_links[slice_id].add_node(b) 
        # slices_links[slice_id].add_edge(a,b, date=datetime_object)   
        slices_links[slice_id].add_edge(a,b, date=tm.mktime(datetime_object.timetuple()))


    for slice_id in slices_links:
        print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
        print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
        
        temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
        print ("Shape of temp matrix", temp.shape)
        slices_features[slice_id] = {}
        for idx, node in enumerate(slices_links[slice_id].nodes()):
            slices_features[slice_id][node] = temp[idx]
    # TODO : remap and output.
    slices_links_remap, slices_features_remap = remap(slices_links, slices_features)

    np.savez('dataset/UCI/raw/graphs.npz', graph=slices_links_remap)
    np.savez('dataset/UCI/raw/features.npz', feats=slices_features_remap)


def from_networkx(G, x, anormaly, group_node_attrs=None,group_edge_attrs = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        anormaly (bool) : whether or not to add anormaly to current graph
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    

    G = nx.convert_node_labels_to_integers(G)
    # G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    data = {}

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}
   
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass
    
    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    
    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = [data[key] for key in group_node_attrs]
        xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
        data.x = torch.cat(xs, dim=-1)
    ##Add feature matrix directly from parameter(Need to Modify Later)
    data.x = torch.from_numpy(x).float()

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        edge_attrs = [data[key] for key in group_edge_attrs]
        edge_attrs = [x.view(-1, 1) if x.dim() <= 1 else x for x in edge_attrs]
        data.edge_attr = torch.cat(edge_attrs, dim=-1)

    #For anormaly injection, random select 1% edges and duplicate it 30 times
    if anormaly:
        edges = data.edge_index.shape[1]
        for i in range(max(1, int(0.2*edges))):
            idx = np.random.randint(edges)
            abnormal_edge_index = data.edge_index[:,idx].view(2,1).expand(2, 30)
            data.edge_index = torch.cat((data.edge_index, abnormal_edge_index), dim=1)
            abnormal_edge_attrs = data.edge_attr[idx].view(1,-1).expand(30, -1)
            data.edge_attr = torch.cat((data.edge_attr,abnormal_edge_attrs), dim=0)
        data['y'] = torch.tensor([1])
    else:
        data['y'] = torch.tensor([0])

    return data
