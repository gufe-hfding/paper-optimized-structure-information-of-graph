import networkx as nx
import twoDimensionalStructuralInformation as td
import readAdjMatrix
from readAdjMatrix import readAdjMatrix
import random
import numpy
import time
import math
from treelib import Node, Tree
from goto import with_goto
import getPartition

def init_codetree(G):
    
    codetree=Tree()
    codetree.create_node(tag='lamda',identifier=-1,data=None)
    for node in list(nx.nodes(G)):
       codetree.create_node(tag=node,identifier=node,parent=-1,data=None)

    return codetree


def ids_of_all_nodes(T):
    """
    ids_of_all_nodes is a function that returns the identifiers of all nodes of T in a list
    param T: a tree
    return: identifiers of all nodes of T 
    """
    ids=[]
    for node in T.all_nodes():
        ids.append(node.identifier)
    return ids

def self_entropy(G,T,alpha):
    """
    self_entropy is a function that estimates the entropy of a node (with id alpha) in a codetree T of graph G
    param G: the given graph
    param T: a codetree of given graph G
    param alpha: the id of a node in T
    return: the self entropy of the node that with id alpha
    """
    if T.get_node(alpha).is_root():
        print("Error paramator in function self_entropy: alpha is illegel.")
        return 0

    parent_id = T.get_node(alpha).bpointer
    leaf_ids_of_node =[]
    leaf_ids_of_parent = []
    for node in T.leaves(alpha):
        leaf_ids_of_node.append(node.identifier)
    for node in T.leaves(parent_id):
        leaf_ids_of_parent.append(node.identifier)

    g_node = nx.cut_size(G, leaf_ids_of_node, list(set(nx.nodes(G)).difference(set(leaf_ids_of_node))))
    v_G = nx.volume(G,nx.nodes(G))
    v_node = nx.volume(G,leaf_ids_of_node)
    v_parent = nx.volume(G,leaf_ids_of_parent)

    entropy = - g_node/v_G * math.log2(v_node/v_parent)
    return entropy

def merge(T,alpha,beta):
    """
    merge is a function that mergers two nodes of tree T, changing node beta to be a child of node alpha, where alpha and beta are sisters.
    param T: the given tree
    param alpha: the target node of T
    param beta: the merged node of T
    return: the new tree after merging operation.
    """
    TT = Tree(T.subtree(T.root), deep=True)
    TT.move_node(beta,alpha)
    return TT

def combine(T, alpha, beta):
    """
    combine is a function that combines two nodes of tree T, changing nodes alpha and beta to be children of a new node gama, where alpha and beta are sisters.
    param T: the given tree
    param alpha: the target node of T
    param beta: the target node of T
    return: the new tree after combining operation.
    """
    TT = Tree(T.subtree(T.root), deep=True)
    new_id = max(ids_of_all_nodes(T))+1
    parent_id = TT.get_node(alpha).bpointer
    TT.create_node(tag=new_id,identifier=new_id,parent=parent_id,data=None)
    TT.move_node(alpha, new_id)
    TT.move_node(beta, new_id)
    return TT

def merge_delta(G, T, TT, alpha, beta):
    """
    merge_delta is a function that estimatis the differenc between the structural information of graph G given by coding tree T and merged coding tree TT of alpha and beta.
    param G: the given graph
    param T: the original coding tree of G
    param TT: the merged coding tree of G by T, with merged nodes alpha and beta
    param alpha: the target node of G given by coding tree T
    param beta: the merged node of G given by coding tree T
    return entropy_delta: the difference entropy before and after the merging operation by function merge(T, alpha, beta)
    """
    entropy_delta = 0

    for node in T.expand_tree(alpha):
        entropy_delta += self_entropy(G,T,T.get_node(node).identifier)
    for node in T.expand_tree(beta):
        entropy_delta += self_entropy(G,T,T.get_node(node).identifier)
    
    for node in TT.expand_tree(alpha):
        entropy_delta -= self_entropy(G,TT,TT.get_node(node).identifier)
    
    return entropy_delta

def combine_delta(G,T,TT,alpha,beta):
    """
    combine_delta is a function that estimatis the differenc between the structural information of graph G given by coding tree T and combined coding tree TT of alpha and beta.
    param G: the given graph
    param T: the original coding tree of G
    param TT: the combined coding tree of G by T, with combined nodes alpha and beta
    param alpha: the target node of G given by coding tree T
    param beta: the combined node of G given by coding tree T
    return entropy_delta: the difference entropy before and after the combining operation by function combine(T, alpha, beta)
    """
    new_parent_id = TT.get_node(alpha).bpointer
    entropy_delta = self_entropy(G,T,alpha) + self_entropy(G,T,beta) - self_entropy(G,TT,new_parent_id) - self_entropy(G,TT,alpha) - self_entropy(G,TT,beta)
    return entropy_delta 

def shannon_entropy(G):
    """
    shannon_entropy is a function that estimates the 1-dimensional strutural entropy (i.e. Shonnon entropy) of graph G
    param G: graph
    return: the entropy and the structure None.
    note: defined by Angsheng Li and Yichen Pan in [1], definition 1.
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    """
    entropy = 0
    vol=nx.volume(G,nx.nodes(G))
    for node in list(nx.nodes(G)):
        p = nx.degree(G,node)/vol
        entropy += -p* math.log2(p)
    return entropy, None


def structural_entropy(G):
    """
    structural_entropy is a function that estimates the 2-dimensional strutural entropy (i.e. structural information) of graph G
    param G: graph
    return: the structural information(entropy) and the structure of underlying entropy (i.e. the partition of G that the entropy is minimum).
    note: defined by Angsheng Li and Yichen Pan in [1], definition 6, this algorithm is implemented by Li et al in [2] and can be accessed at [3].
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    [2] Angsheng L , Xianchen Y , Bingxiang X , et al. Decoding topologically associating domains with ultra-low resolution Hi-C data by graph structural entropy. Nature Communications, 2018, 9(1):3265-.
    [3] https://github.com/yinxc/structural-information-minimisation
    """
    codetree = structural_coding_tree(G)
    entropy = 0
    for idx in ids_of_all_nodes(codetree)[1:]:
        entropy += self_entropy(G,codetree,idx)
    return entropy, codetree


@with_goto
def structural_coding_tree(G):
    """
    structural_entropy is a function that estimates the 2-dimensional strutural entropy (i.e. structural information) of graph G
    param G: graph
    return: the structural information(entropy) and the structure of underlying entropy (i.e. the partition of G that the entropy is minimum).
    note: defined by Angsheng Li and Yichen Pan in [1], definition 6, this algorithm is implemented by Li et al in [2] and can be accessed at [3].
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    [2] Angsheng L , Xianchen Y , Bingxiang X , et al. Decoding topologically associating domains with ultra-low resolution Hi-C data by graph structural entropy. Nature Communications, 2018, 9(1):3265-.
    [3] https://github.com/yinxc/structural-information-minimisation
    """
    codetree = init_codetree(G)
    loop = 0

    label .merge_operation
    while True:
        fpointers_ids_of_root=codetree.get_node(codetree.root).fpointer
        merge_delta_entropy = 0
        for node_id_alpha in fpointers_ids_of_root:
            alpha_index = fpointers_ids_of_root.index(node_id_alpha)
            for node_id_beta in fpointers_ids_of_root[alpha_index+1:]:
                if not codetree.get_node(node_id_alpha).is_leaf() and codetree.get_node(node_id_beta).is_leaf():
                    mg_T = merge(codetree,node_id_alpha,node_id_beta)
                    ent = merge_delta(G, codetree,mg_T,node_id_alpha, node_id_beta)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)
                elif codetree.get_node(node_id_alpha).is_leaf() and not codetree.get_node(node_id_beta).is_leaf():
                    mg_T = merge(codetree,node_id_beta,node_id_alpha)
                    ent = merge_delta(G, codetree,mg_T, node_id_beta, node_id_alpha)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)
        if merge_delta_entropy > 0:
            codetree = merged_tree
        else:
            break
    
    label .combine_operation
    while True:
        fpointers_ids_of_root=codetree.get_node(codetree.root).fpointer
        combine_delta_entropy = 0
        for node_id_alpha in fpointers_ids_of_root:
            alpha_index = fpointers_ids_of_root.index(node_id_alpha)
            for node_id_beta in fpointers_ids_of_root[alpha_index+1:]:
                if codetree.get_node(node_id_alpha).is_leaf() and codetree.get_node(node_id_beta).is_leaf():
                    cm_T = combine(codetree,node_id_alpha,node_id_beta)
                    ent = combine_delta(G, codetree,cm_T,node_id_alpha, node_id_beta)
                    if ent > combine_delta_entropy:
                        combine_delta_entropy = ent
                        combined_tree = Tree(cm_T.subtree(cm_T.root), deep=True)
        if combine_delta_entropy > 0:
            codetree = combined_tree
        else:
            loop += 1
            break
    if loop > 2: 
        return codetree
    else:
        goto .merge_operation

def optimized_structural_entropy(G):
    """
    optimized_structural_entropy is a function that estimates the 2-dimensional strutural entropy (i.e. structural information) of graph G
    param G: graph
    return: the structural information(entropy) and the structure of underlying entropy (i.e. the partition of G that the entropy is minimum).
    note: defined by Angsheng Li and Yichen Pan in [1], definition 6.
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    """
    codetree = optimized_structural_coding_tree(G)
    entropy = 0
    for idx in ids_of_all_nodes(codetree)[1:]:
        entropy += self_entropy(G,codetree,idx)
    return entropy, codetree


def optimized_structural_coding_tree(G):
    """
    optimized_structural_coding_tree is a function that estimates the 2-dimensional strutural entropy (i.e. structural information) of graph G
    param G: graph
    return: the structural information(entropy) and the structure of underlying entropy (i.e. the partition of G that the entropy is minimum).
    note: we modify the algorithm, which is defined by Angsheng Li and Yichen Pan in [1], definition 6, and implemented by Li et al in [2].
    [1] Angsheng Li, Yicheng Pan: Structural Information and Dynamical Complexity of Networks. IEEE Trans. Information Theory 62(6): 3290-3339 (2016)
    [2] Angsheng L , Xianchen Y , Bingxiang X , et al. Decoding topologically associating domains with ultra-low resolution Hi-C data by graph structural entropy. Nature Communications, 2018, 9(1):3265-.
    """
    codetree = init_codetree(G)
    while True:
        if codetree.depth() > nx.number_of_nodes(G):
            break
        fpointers_ids_of_root=codetree.get_node(codetree.root).fpointer
        merge_delta_entropy = 0
        for node_id_alpha in fpointers_ids_of_root:
            alpha_index = fpointers_ids_of_root.index(node_id_alpha)
            for node_id_beta in fpointers_ids_of_root[alpha_index+1:]:
                if not codetree.get_node(node_id_alpha).is_leaf() and codetree.get_node(node_id_beta).is_leaf():
                    mg_T = merge(codetree,node_id_alpha,node_id_beta)
                    ent = merge_delta(G, codetree,mg_T,node_id_alpha, node_id_beta)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)
                elif codetree.get_node(node_id_alpha).is_leaf() and not codetree.get_node(node_id_beta).is_leaf():
                    mg_T = merge(codetree,node_id_beta,node_id_alpha)
                    ent = merge_delta(G, codetree,mg_T, node_id_beta, node_id_alpha)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)


        combine_delta_entropy = 0
        for node_id_alpha in fpointers_ids_of_root:
            alpha_index = fpointers_ids_of_root.index(node_id_alpha)
            for node_id_beta in fpointers_ids_of_root[alpha_index+1:]:
                if codetree.get_node(node_id_alpha).is_leaf() and codetree.get_node(node_id_beta).is_leaf():
                    cm_T = combine(codetree,node_id_alpha,node_id_beta)
                    ent = combine_delta(G, codetree, cm_T, node_id_alpha, node_id_beta)
                    if ent > combine_delta_entropy:
                        combine_delta_entropy = ent
                        combined_tree = Tree(cm_T.subtree(cm_T.root), deep=True)
         
        if merge_delta_entropy > 0 and merge_delta_entropy > combine_delta_entropy:
            codetree = Tree(merged_tree.subtree(merged_tree.root), deep=True)
        elif combine_delta_entropy > 0 and combine_delta_entropy > merge_delta_entropy:
            codetree = Tree(combined_tree.subtree(combined_tree.root), deep=True)
        else:
            break

    return codetree


def deDoc(G,k):
    return 0,None

def optimal_graph_coding_tree(G):
    codetree = init_codetree(G)
    while True:
        # print("round:")
        merge_delta_entropy = 0
        for alpha_nid in codetree.expand_tree(mode = 2):
            for beta_node in codetree.siblings(alpha_nid):
                if not codetree.get_node(alpha_nid).is_leaf() and beta_node.is_leaf():    
                    mg_T = merge(codetree, alpha_nid, beta_node.identifier)
                    ent = merge_delta(G, codetree, mg_T, alpha_nid, beta_node.identifier)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)
                        # print("merge tree:")
                        # print("en:",merge_delta_entropy)
                        # merged_tree.show()
                elif codetree.get_node(alpha_nid).is_leaf() and not beta_node.is_leaf():
                    mg_T = merge(codetree, beta_node.identifier, alpha_nid)
                    ent = merge_delta(G, codetree, mg_T, beta_node.identifier, alpha_nid)
                    if ent > merge_delta_entropy:
                        merge_delta_entropy = ent
                        merged_tree = Tree(mg_T.subtree(mg_T.root), deep=True)
                        # print("merge tree:")
                        # print("en:",merge_delta_entropy)
                        # merged_tree.show()
       
        combine_delta_entropy = 0
        for alpha_nid in codetree.expand_tree(nid = -1, mode = 2):
            for beta_node in codetree.siblings(alpha_nid):
                if len(codetree.siblings(alpha_nid)) == 1:
                    break 
                if codetree.get_node(alpha_nid).is_leaf() and beta_node.is_leaf():
                    cm_T = combine(codetree, alpha_nid, beta_node.identifier) 
                    ent = combine_delta(G, codetree, cm_T, alpha_nid, beta_node.identifier)
                    if ent > combine_delta_entropy:
                        combine_delta_entropy = ent
                        combined_tree = Tree(cm_T.subtree(cm_T.root), deep=True)
                        # print("combine tree:")
                        # print("en:",combine_delta_entropy)
                        # combined_tree.show()
         
        if merge_delta_entropy > 0 and merge_delta_entropy > combine_delta_entropy:
            codetree = Tree(merged_tree.subtree(merged_tree.root), deep=True)
        elif combine_delta_entropy > 0 and combine_delta_entropy > merge_delta_entropy:
            codetree = Tree(combined_tree.subtree(combined_tree.root), deep=True)
        else:
            break
    return codetree

def optimal_graph_entropy(G):
    codetree = optimal_graph_coding_tree(G)
    ids_of_codetree= ids_of_all_nodes(codetree)
    entropy=0
    for idx in ids_of_codetree[1:]:
        entropy += self_entropy(G, codetree, idx)
    return entropy, codetree


def k_dimensional_entropy(G,k):
    if k<1:
        print("Illegal k value for estimating the k-dimensional structural entropy of G.") 
    elif k==1:
        return shannon_entropy(G)
    elif k==2:
        return structural_entropy(G)
    else:
        return deDoc(G,k)

def optimal_structural_entropy(G):
    optimal_entropy=float('inf')
    for k in range(1,nx.number_of_nodes(G)):
        k_entropy_and_tree=k_dimensional_entropy(G,k)
        if optimal_entropy > k_entropy_and_tree[0]:
            optimal_entropy=k_entropy_and_tree[0]
            optimal_codetree=k_entropy_and_tree[1]
    return optimal_entropy, optimal_codetree       



if __name__ == "__main__": 
    numpy.random.seed(int(time.time()))
    n=numpy.random.randint(30,35)
    p=numpy.random.uniform(0.3,0.7)

    G=nx.fast_gnp_random_graph(n,p,int(time.time()))
    subG=nx.k_core(G,1)

    # adjmatrix=readAdjMatrix("./data/matrix")
    # subG=nx.from_numpy_matrix(adjmatrix,False,None)
    print(nx.nodes(subG))
    print(nx.edges(subG))
    # initial_structure = ((),(),(),())
    # codetree = nx.from_nested_tuple(initial_structure, sensible_relabeling=True)
    # print(nx.nodes(codetree))
    # print(nx.edges(codetree))

    en, tr = shannon_entropy(subG)
    print("Shannon entropy:", en)
    en,tree = structural_entropy(subG)
    print("en:", en)
    tree.show()

    print("Optimized structural entropy:")
    en,tree = optimized_structural_entropy(subG)
    print("en:", en)
    tree.show()

    print("Optimal graph strcutural entropy:")
    en,tree = optimal_graph_entropy(subG)
    print("en:", en)
    tree.show()

    # print("The 2 dimensional structural information and corresponding partition of G are:")
    # print(td.twoDimensionalStructuralInformation(subG))
    # print("The optimal 2 dimensional structural information and core based partition of G are:")
    # print(td.optimalTwoDimensionalStructuralInformation(subG))
    # adjm=nx.to_numpy_matrix(subG)
    # numpy.savetxt('matrix.csv', adjm, delimiter = ',') 

    # part=[[0,1],[2,3],[6,7,8],[9,10,11],[12,13,14,15],[19,20,21,22],[23,24,25,26,27,28],[5],[6],[17],[18],[19]]
    # print(td.structuralInformation(subG,part))



    # codetree=Tree()
    # codetree.create_node(tag='lamda',identifier=-1,data=None)
    # for node in list(nx.nodes(subG)):
    #    codetree.create_node(tag=node,identifier=node,parent=-1,data=None)
    # print(codetree.nodes)
    # codetree.show()
    # T = merge(codetree,2,3)
    # T = combine(codetree,4,5)
    # codetree.show()
    # T.show()
    # print(T.siblings(4))

    # ids = ids_of_all_nodes(codetree)
    # print(codetree.root)
    # print(ids.remove(-1))
    # print(codetree.leaves())
    # leaves_list=[]
    # for item in codetree.leaves():
    #     leaves_list.append(item.tag)
    # print(leaves_list)

    # print(codetree.get_node(-1).fpointer)






