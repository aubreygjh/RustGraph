#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt  
from torch_geometric.utils import homophily, assortativity, to_networkx
from dataset import  DynamicGraphAnomaly

'''
# #visualize graphs
# for i,graph in enumerate(data):
#     draw(graph.edge_index, graph.y, i)

# def draw(edge_index, y, name=None):
#     G = nx.MultiGraph(node_size=15, font_size=8)
#     src = edge_index[0].cpu().numpy()
#     dst = edge_index[1].cpu().numpy()
#     edgelist = zip(src, dst)
#     for i, j in edgelist:
#         G.add_edge(i, j)
#     plt.figure(figsize=(20, 14)) # 设置画布的大小
#     if y == 1:
#         nx.draw_networkx(G,node_color="red")
#     else:
#         nx.draw_networkx(G,node_color="blue")
#     if not os.path.exists('figs'):
#         os.mkdir('figs')
#     plt.savefig('figs/{}.png'.format(name if name else 'path'))
#     print(f'Saved fig-{name}.')
'''



def plot_tsne(edge_emb, y, epoch, t):
    # Normalize edge_emb
    edge_emb_normalized = (edge_emb - edge_emb.mean(dim=0)) / edge_emb.std(dim=0)

    # Apply T-SNE
    tsne = TSNE(n_components=2)
    edge_emb_tsne = tsne.fit_transform(edge_emb_normalized.detach().cpu().numpy())

    # Get the labels
    classes = y.detach().cpu().numpy() 
    unique_classes = np.unique(classes)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(edge_emb_tsne[:, 0], edge_emb_tsne[:, 1], c=classes)

    plt.title("T-SNE Visualization of edge_emb")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # Create a legend with labels
    plt.legend()

    plt.savefig(f'tsne_{epoch}_{t}.pdf')
    plt.show()



def get_properties(datasets):
    # Empty lists to store clustering coefficients and densities for each dataset
    dataset_clustering_coeffs = []
    dataset_densities = []
    dataset_assortativity_coeffs = []
    dataset_transitivity = []
    dataset_connected_components = []
    dataset_degrees = []
    dsnames = ['UCI', 'Digg', 'BTC-Alpha', 'BTC-OTC', 'Email', 'AS-Topology']

    # Iterate over each dataset
    for dataset in datasets:
        # Empty lists to store clustering coefficients and densities for the current dataset
        clustering_coeffs = []
        densities = []
        assortativity_coeffs = []
        transitivity = []
        connected_components = []
        degrees = []

        # Iterate over each graph in the current dataset
        for data in dataset:
            # Convert to NetworkX graph
            graph = to_networkx(data)

            # Compute clustering coefficient
            clustering_coefficient = nx.average_clustering(graph)
            clustering_coeffs.append(clustering_coefficient)

            # Compute density
            density = nx.density(graph)
            densities.append(density)

            # Compute assortativity coefficient
            assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
            assortativity_coeffs.append(assortativity_coefficient)

            # Compute transitivity
            transitivity_coefficient = nx.transitivity(graph)
            transitivity.append(transitivity_coefficient)

            # Compute number of connected components
            connected_components.append(nx.number_connected_components(graph.to_undirected()))


            # Calculate the average degree
            degree = dict(nx.degree(graph))
            average_degree = sum(degree.values()) / len(degree)
            degrees.append(average_degree)
        print(np.mean(connected_components))

        # Store the clustering coefficients, densities, assortativity coefficients, and transitivity for the current dataset
        dataset_clustering_coeffs.append(clustering_coeffs)
        dataset_densities.append(densities)
        dataset_assortativity_coeffs.append(assortativity_coeffs)
        dataset_transitivity.append(transitivity)
        dataset_connected_components.append(connected_components)
        dataset_degrees.append(degrees)

    # Plot the clustering coefficient, density, assortativity, and transitivity curves in the same figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, ax in enumerate(axs.flat):        
        # Plot the clustering coefficient curves for each dataset
        if i == 0:
            ax.set_xlabel('Snapshots')
            ax.set_ylabel('Clustering Coefficient')
            ax.set_title('Clustering Coefficient Curves', fontsize=10)
            for j, clustering_coeffs in enumerate(dataset_clustering_coeffs):
                line_style = line_styles[j % len(line_styles)]
                marker = markers[j % len(markers)]  # Use a different marker for each curve
                color = curve_colors[j % len(curve_colors)]
                ax.plot(range(min(len(clustering_coeffs), 50)), clustering_coeffs[:50], label=dsnames[j], linestyle=line_style, color=color)
                ax.scatter([0, len(clustering_coeffs[:50]) - 1], [clustering_coeffs[0], clustering_coeffs[len(clustering_coeffs[:50]) - 1]], marker=marker, color=color, label=None)
        
        # Plot the density curves for each dataset
        elif i == 1:
            ax.set_xlabel('Snapshots')
            ax.set_ylabel('Density')
            ax.set_title('Density Curves', fontsize=10)
            for j, densities in enumerate(dataset_densities):
                line_style = line_styles[j % len(line_styles)]
                marker = markers[j % len(markers)]  # Use a different marker for each curve
                color = curve_colors[j % len(curve_colors)]
                ax.plot(range(min(len(densities), 50)), densities[:50], label=dsnames[j], linestyle=line_style, color=color)
                ax.scatter([0, len(densities[:50]) - 1], [densities[0], densities[len(densities[:50]) - 1]], marker=marker, color=color, label=None)

        # Plot the transitivity curves for each dataset
        elif i == 2:
            ax.set_xlabel('Snapshots')
            ax.set_ylabel('Transitivity')
            ax.set_title('Transitivity Curves', fontsize=10)
            for j, transitivity in enumerate(dataset_transitivity):
                line_style = line_styles[j % len(line_styles)]
                marker = markers[j % len(markers)]  # Use a different marker for each curve
                color = curve_colors[j % len(curve_colors)]
                ax.plot(range(min(len(transitivity), 50)), transitivity[:50], label=dsnames[j], linestyle=line_style, color=color)
                ax.scatter([0, len(transitivity[:50]) - 1], [transitivity[0], transitivity[len(transitivity[:50]) - 1]], marker=marker, color=color, label=None)

        # Plot the Connected Components curves for each dataset
        elif i == 3:
            ax.set_xlabel('Snapshots')
            ax.set_ylabel('Connected Components')
            ax.set_title('Connected Components Curves', fontsize=10)
            for j, connected_components in enumerate(dataset_connected_components):
                line_style = line_styles[j % len(line_styles)]
                marker = markers[j % len(markers)]  # Use a different marker for each curve
                color = curve_colors[j % len(curve_colors)]
                ax.plot(range(min(len(connected_components), 50)), connected_components[:50], label=dsnames[j], linestyle=line_style, color=color)
                ax.scatter([0, len(connected_components[:50]) - 1], [connected_components[0], connected_components[len(connected_components[:50]) - 1]], marker=marker, color=color, label=None)

        # elif i == 3:
        #     ax.set_xlabel('snapshots')
        #     ax.set_ylabel('degrees')
        #     ax.set_title('degrees',fontsize=10)
        #     for j, assortativity_coefficient in enumerate(dataset_degrees):
        #         line_style = line_styles[j % len(line_styles)]
        #         marker = markers[j % len(markers)]  # Use a different marker for each curve
        #         color = curve_colors[j % len(curve_colors)]
        #         ax.plot(range(min(len(assortativity_coefficient),50)), assortativity_coefficient[:50],label=dsnames[j],linestyle=line_style,color=color)
                

    # Add a legend to the first subplot
    axs[1][1].legend()
    # Adjust the spacing between subplots
    plt.tight_layout()
    # Save the figure as a PDF file in the current directory
    plt.savefig('time_vairant_properties.pdf')
    # Display the figure
    plt.show()
    

def visualize_topology(datasets):
    # Iterate over each dataset
    dsnames = ['Digg', 'AS-Topology'] #'UCI Messages', 'Bitcoin-Alpha', 'Bitcoin-OTC', 'Emain-DNC'

    for i, dataset in enumerate(datasets):
        dataset_name = dsnames[i]

        # Get the first, middle, and last snapshots from the dataset
        num_snapshots = len(dataset)
        first_snapshot = dataset[0]
        middle_snapshot = dataset[num_snapshots // 2]
        last_snapshot = dataset[-1]
        
        # Convert to NetworkX graphs
        first_graph = to_networkx(first_snapshot)
        middle_graph = to_networkx(middle_snapshot)
        last_graph = to_networkx(last_snapshot)
 
        # Create subplots for the topology visualization
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first snapshot
        axs[0].set_title(f"{dataset_name} - First Snapshot")
        nx.draw(first_graph, ax=axs[0], node_size=10, node_color='blue', edge_color='gray', with_labels=False)

        # Plot the middle snapshot
        axs[1].set_title(f"{dataset_name} - Middle Snapshot")
        nx.draw(middle_graph, ax=axs[1], node_size=10, node_color='blue', edge_color='gray', with_labels=False)

        # Plot the last snapshot
        axs[2].set_title(f"{dataset_name} - Last Snapshot")
        nx.draw(last_graph, ax=axs[2], node_size=10, node_color='blue', edge_color='gray', with_labels=False)

        # Adjust the spacing between subplots
        plt.tight_layout()
        # Save the figure as a PDF with the name of the corresponding dataset
        plt.savefig(f"{dataset_name}_topology.pdf")
        # Display the figure
        plt.show()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','y','true','t','1'):
        return True
    if v.lower() in ('no','n','false','f','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

    # ===== dataset parameters =====
    parser.add_argument('--dataset', type=str, default='uci', help='name of dataset')
    parser.add_argument('--snap_size', type=int, default=500, help='')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

    # ===== training parameters =====
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    # ===== hyper-parameters =====
    parser.add_argument('--window', type=int, default=1, help='')
    parser.add_argument('--eps', type=float, default=0.2, help='eps')
    parser.add_argument('--bce_weight', type=float, default=1, help='')
    parser.add_argument('--gen_weight', type=float, default=1, help='')
    parser.add_argument('--con_weight', type=float, default=1, help='')
    parser.add_argument('--reg_weight', type=float, default=1, help='')

    # ===== model parameters =====
    parser.add_argument('--layer_num', type=int, default=2, help='rnn layers')
    parser.add_argument('--x_dim', type=int, default=256, help='input channels of the model')
    parser.add_argument('--h_dim', type=int, default=256, help='hidden channels of the model')
    parser.add_argument('--z_dim', type=int, default=256, help='output channels of the model')

    # ===== may remove in the future =====
     # parser.add_argument('--hidden_dim_rnn', type=int, default=256, help='hidden channels of rnn')
    # parser.add_argument('--num_nodes', type=int, default=1809, help='number of all nodes')
    # parser.add_argument('--num_layers_gae', type=int, default=2, help='gnn layers')
    # parser.add_argument('--input_dim_gae', type=int, default=1809, help='input channels of gae')
    # parser.add_argument('--hidden_dim_gae', type=int, default=512, help='hidden channels of gae')
    # parser.add_argument('--add_ev', type=str2bool, default=True, help='add_ev')
    # parser.add_argument('--timespan', type=int, default=6, help='k timestamp for contrastive learning')
    # parser.add_argument('--num_classes', type=int, default=2, help='normal or abnormal')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Device      : {args.device}')
    print(f'    Learning Rate   : {args.lr}')
    print(f'    Training Epochs : {args.epochs}')

    print('\n Network details:')
    print(f'    1st-GNN in: {args.in_channels_gnn}, out:{args.hidden_channels_gnn}')
    print(f'    2nd-GNN in: {args.hidden_channels_gnn}, out:{args.out_channels_gnn}')
    print(f'    Eigenvector as representation: {args.add_ev}')
    if args.add_ev:
        print(f'    RNN in {args.out_channels_gnn + args.num_nodes}, out:{args.out_channels_gnn + args.num_nodes}')
    else:
        print(f'    RNN in {args.out_channels_gnn}, out:{args.out_channels_gnn}')
    print(f'    N_samples:{args.n_samples}')
    print(f'    Timestamp:{args.timestamp}\n')

    return

if __name__ == '__main__':
    #Init dataloader
    args = args_parser()
    datasets = ['uci', 'digg', 'btc_alpha', 'btc_otc', 'email', 'as_topology'] # ，
    graphs = []
    for dsname in datasets:
        if dsname == 'email':
            args.x_dim = 512
        elif dsname == 'as_topology':
            args.x_dim = 64
        else:
            args.x_dim = 128
        dataset = DynamicGraphAnomaly(root='dataset', name=dsname, args=args)
        graphs.append(dataset[:])

    get_properties(graphs)
    # visualize_topology(graphs)

    # run this command to visualize:
    # CUDA_VISIBLE_DEVICES=5 python utils.py  --snap_size 1000 --train_ratio 0.5 --anomaly_ratio 0.1 --noise_ratio 0  --epoch 200 --lr 0.001 --x_dim 128 --h_dim 128 --z_dim 128 