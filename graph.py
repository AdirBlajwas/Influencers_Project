# Read the graph list from the pickle file
import networkx as nx
import pandas as pd
import pickle
import random
import numpy as np
from simulation_generator import create_graph


def edge_probability(G,u,v):
    """
    :return: the probability of an edge being added between u and v
    """
    common_neighbors = set(G.neighbors(u)).intersection(set(G.neighbors(v)))
    k_expectation = sum(G[u][w]['weight'] * G[v][w]['weight'] for w in common_neighbors)
    p = 0.01
    if k_expectation < 1:
        return 0
    return 1 - (1-p)**np.log(k_expectation)

def create_weighted_G(G):
    """
    Initialize a wighted graph with all edges having weight 1
    :param G: original unweighted graph
    :return: weighted graph
    """
    for u, v in G.edges():
        if G.has_edge(u, v):
            G[u][v]['weight'] = 1
    return G


def create_weighted_graphs(G, p, num_periods, threshold):
    """
    Create a list of weighted graphs, where each graph corresponds to a period of time.
    :param G: graph in period 0
    :param p: edge probability function
    :param num_periods: number of periods
    :param threshold: minimum weight for an edge to be added
    """
    # Iterate through all periods of time.
    weighted_graphs = [create_weighted_G(G)]
    for t in range(num_periods):
        edges_to_add = []
        # Create a copy of the original graph G.
        G_t = weighted_graphs[t].copy()
        # Iterate through all edges in G.
        for u in sorted(G_t.nodes):
            for v in sorted(G_t.nodes,reverse=True):
                # Check if the edge exists in the copy of the graph corresponding to the current period of time.
                if u == v:
                    break
                prob = 0
                if (u, v) not in G_t.edges:
                    # If the edge does not exist, assign a weight equal to the probability of that edge being added by the specified time.
                    prob = p(G_t, u, v)
                elif G_t[u][v]['weight'] < 1:
                    cur_weight = G_t[u][v]['weight']
                    prob = 1 - (1-cur_weight)*(1-p(G_t, u, v))
                if prob > threshold:
                    edges_to_add.append((u, v, prob))
        G_t.add_weighted_edges_from(edges_to_add)
        # Append the weighted graph for the current period of time to the list of weighted graphs.
        print("----------------")
        print(G_t)
        weighted_graphs.append(G_t)

    return weighted_graphs


def create_next_graph(G, p, round):
    count = 0
    edges_to_add = []
    # Create a copy of the original graph G.
    G_t = G.copy()
    # Iterate through all edges in G.
    for u in sorted(G_t.nodes):
        for v in sorted(G_t.nodes, reverse=True):
            print(count)
            count += 1
            # Check if the edge exists in the copy of the graph corresponding to the current period of time.
            if u == v:
                break
            if (u, v) not in G_t.edges or G_t[u][v]['weight'] < 1:
                # If the edge does not exist, assign a weight equal to the probability of that edge being added by the specified time.
                prob = p(G_t, u, v)
                if prob > 0.05:
                    edges_to_add.append((u, v, prob))
    G_t.add_weighted_edges_from(edges_to_add)
    # Append the weighted graph for the current period of time to the list of weighted graphs.
    print("----------------")
    print(G_t)

    with open(f"weighted_graph{round}.pkl", 'wb') as f:
        pickle.dump(G_t, f)




def remove_edges_below_threshold(G, threshold):
    """
    Removes all edges with weight below the threshold from the graph G.
    :param G: weighted graph
    :param threshold: threshold for edge weight
    :return: cleaned graph
    """
    # Create a new graph with the same nodes as G
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Create a list of edges to add to the new graph
    edges_to_add = [(u, v, weight) for u, v, weight in G.edges(data='weight') if weight >= threshold]

    # Add the edges to the new graph
    H.add_weighted_edges_from(edges_to_add)

    # Return the new graph
    return H

def print_Graphs(G_list):
    for i in range(7):
        print("------------ -")
        print(G_list[i])


def main():
    n = 6 # number of periods
    threshold = 0.1 # threshold for edge weight
    G = create_graph('chic_choc_data.csv')
    G = create_weighted_G(G)
    print("done creating graph")
    weighted_graphs = create_weighted_graphs(G, edge_probability, n, 0.05)
    weighted_graphs = [remove_edges_below_threshold(G, threshold) for G in weighted_graphs]
    print_Graphs(weighted_graphs)
    with open(f'weighted_graphs_cleaned.pkl', 'wb') as g:
        pickle.dump(weighted_graphs,g)
main()
