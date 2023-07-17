import numpy as np
import networkx as nx
import random
import pandas as pd
import pickle
from choose_influencers import find_best_influencers


p = 0.01
chic_choc_path = 'chic_choc_data.csv'
cost_path = 'costs.csv'


def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the Chic Choc social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The Chic Choc social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net


def change_network(net: nx.Graph) -> nx.Graph:
    """
    Gets the network at staged t and returns the network at stage t+1 (stochastic)
    :param net: The network at staged t
    :return: The network at stage t+1
    """
    edges_to_add = []
    for user1 in sorted(net.nodes):
        for user2 in sorted(net.nodes, reverse=True):
            if user1 == user2:
                break
            if (user1, user2) not in net.edges:
                neighborhood_size = len(set(net.neighbors(user1)).intersection(set(net.neighbors(user2))))
                prob = 1 - ((1 - p) ** (np.log(neighborhood_size))) if neighborhood_size > 0 else 0  # #################
                if prob >= random.uniform(0, 1):
                    edges_to_add.append((user1, user2))
    net.add_edges_from(edges_to_add)
    return net



def buy_products(net: nx.Graph, purchased: set) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param net: The network at stage t
    :param purchased: All the users who bought a doll up to and including stage t-1
    :return: All the users who bought a doll up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)

def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])


def simulate(influencers):
    print("STARTING")

    chic_choc_network = create_graph(chic_choc_path)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    purchased = set(influencers)

    for i in range(6):
        chic_choc_network = change_network(chic_choc_network)
        purchased = buy_products(chic_choc_network, purchased)
        print("finished round", i + 1)

    score = len(purchased) - influencers_cost

    print("*************** Your final score is " + str(score) + " ***************")
    return score


if __name__ == '__main__':
    influencers = None
    with open('influencers.pkl', 'rb') as g:
        influencers = pickle.load(g)
    score = 0
    n = 20
    for i in range(n):
        score += simulate(influencers)
    print(f"{influencers} avg score out of {n} is {score / n}")
