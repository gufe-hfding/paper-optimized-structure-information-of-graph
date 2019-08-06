"""
author: Hongfa Ding
finish date: 2019-7-21
"""
import sys

import matplotlib.pyplot as plt
import networkx as nx
import readAdjMatrix
from readAdjMatrix import readAdjMatrix
import getPartition
from getPartition import partition
import math


def getCorePartition(G):
    """
    getCorePartitioin is a generator that generates the core based partition of a given graph.
    :param G: a graph.
    :return: the core based partition propsed by Ding et al. (ourself's paper)
    """
    delta = max(nx.core_number(G).values())
    corenodes = []
    for i in range(delta):
        corenodes.append(list(nx.k_core(G, i+1).nodes()))
    partition = corenodes
    for i in range(0, delta-1):
        partition[i] = list(
            set(partition[i]).difference(set(partition[i+1])))
    core_partition=[]
    for i in range(delta):
        core_subgraph=nx.connected_components(nx.induced_subgraph(G,partition[i]))
        for sg in core_subgraph:
            core_partition.append(list(sg))
    return core_partition

def structuralInformation(G, P):
    """
    structuralInformation is a function that estimates the structural information of graph G by a partition P of G.
    :param G: a graph.
    :param P: a partition of G.
    :return: the structural information of G defined by Li and Pan in [1], definition 6.
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    """
    G_volume=nx.volume(G,nx.nodes(G))
    # print(G_volume)
    all_nodes=list(nx.nodes(G))
    #print(all_nodes)
    entropy=0
    for item in P:
        v = nx.volume(G,item)
        g = nx.cut_size(G, item, list(set(all_nodes).difference(set(item))))
        v_entropy=0
        for it in item:
            it_p = nx.degree(G,it)/v
            v_entropy +=  -it_p * math.log(it_p,2)
        entropy += v/G_volume * v_entropy - g/G_volume * math.log(v/G_volume,2) 
    return entropy


def twoDimensionalStructuralInformation(G):
    """
    twoDimensionalStructuralInformation is a function that estimates the 2 dimensional  structural information of graph G.
    :param G: a graph.
    :return: the 2 dimensional structural information of G defined by Li and Pan in [1], definition 9.
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    """
    min_entropy=float('inf')
    max_entropy=0.0
    min_partition=[]
    max_partition=[]
    for idx, item in enumerate(partition(list(nx.nodes(G)))):
        # print("Estimating the", idx, "-th partition of G......")
        current_entropy = structuralInformation(G,item)
        # print(current_entropy,item)
        if min_entropy > current_entropy:
            min_entropy = current_entropy
            min_partition = item
    #     if max_entropy < current_entropy:
    #         max_entropy = current_entropy
    #         max_partition = item
    # print("The maximue entropy is：")
    # print(max_entropy,max_partition)
            
    return min_entropy, min_partition

def optimalTwoDimensionalStructuralInformation(G): 
    """
    optimalTwoDimensionalStructuralInformation is a function that estimates the optimal 2 dimensional  structural information of graph G by core based partition.
    :param G: a graph.
    :return: the optimal 2 dimensional structural information of G defined by Ding et al. (ourself's papar)
    """
    core_partition=getCorePartition(G)
    entropy = structuralInformation(G,core_partition)
    return entropy, core_partition




if __name__ == "__main__":
    adjmatrix=readAdjMatrix("./data/matrix")
    G=nx.from_numpy_matrix(adjmatrix,False,None)
    # print("The graph G is:")
    # nx.draw(G)
    # plt.show()

    # all_nodes = nx.nodes(G)
    # item=[12,13,14,15]
    # g = nx.cut_size(G, item, list(set(all_nodes).difference(set(item))))
    # print(g)

    # all_nodes=nx.nodes(G)
    # i=0
    # for idx, item in enumerate(partition(list(all_nodes)), 1):
    #     print(item)
    #     current_entropy = structuralInformation(G,list(item))
    #     print(current_entropy)
    #     i+=1
    #     if i > 10:
    #         break


    print("The optimal 2 dimensional structural information and core based partition of G are:")
    print(optimalTwoDimensionalStructuralInformation(G))

    print("The 2 dimensional structural information and corresponding partition of G are:")
    print(twoDimensionalStructuralInformation(G))
       