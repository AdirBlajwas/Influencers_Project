import pandas as pd
import numpy as np
import networkx as nx
import pickle
from itertools import combinations

def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])

def IC(weighted_graphs, selected):
    """
    Calculate the influence cone of a given set of influencers
    :param weighted_graphs: the graphs of the n time steps
    :param selected: the given set of influencers
    :return: the influence cone of the given set of influencers
    """
    ip = np.zeros((len(weighted_graphs[0].nodes), 7))
    for u in selected:
        for j in range(1,7):
            ip[u][j] = 1
    for t in range(1, 7):
        G = weighted_graphs[t]
        for v in G.nodes():
            if ip[v][t] == 1:
                continue
            infected_neighbors = 0
            all_neighbors = 0
            for u in G.neighbors(v):
                weight = G[u][v]['weight']
                infected_neighbors += ip[u][t-1] * weight
                all_neighbors += weight
            ip[v][t] = ip[v][t-1] + (1-ip[v][t-1])*(infected_neighbors/all_neighbors)
    return sum(ip[:, 6]) - get_influencers_cost('costs.csv', selected)


def hill_climb_ver(vertices, weighted_graphs):
    """
    Find the best set of influencers using hill climb algorithm (greedy algorithm)
    :param vertices: the vertices of the graph which are possible influencers
    :param weighted_graphs: graphs of the n time steps representing dinamic social network
    :return: the best set of influencers and the influence cone of the set
    """
    print("STARTED")
    s = []
    max_IC = 0
    for t in range(5):
        best_v = None
        for v in vertices:
            v_IC = IC(weighted_graphs, s + [v])
            if v_IC > max_IC:
                print(f"old IC ={max_IC}, s = {s}")
                best_v = v
                max_IC = v_IC
                print(f"new IC ={max_IC}, s = {s + [best_v]}")
        s.append(best_v)
    return s, max_IC


def find_best_influencers():
    """
    Find the best set of influencers using hill climb algorithm (greedy algorithm)
    """
    print("STARTING hill climb")
    with open('weighted_graphs_cleaned.pkl', 'rb') as f:
        weighted_graphs = pickle.load(f)
        influencers, high_IC = hill_climb_ver(list(weighted_graphs[0].nodes), weighted_graphs)
        print(f"best IC = {high_IC}, influencers = {influencers}")
        with open('influencers.pkl', 'wb') as g:
            pickle.dump(influencers, g)
        return influencers

find_best_influencers()