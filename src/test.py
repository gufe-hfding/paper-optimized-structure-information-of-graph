import twoDimensionalStructuralInformation as td 
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy
import time
import scipy
import readAdjMatrix
from readAdjMatrix import readAdjMatrix
numpy.random.seed(int(time.time()))
n=numpy.random.randint(7,9)
p=numpy.random.uniform(0.3,0.7)

G=nx.fast_gnp_random_graph(n,p,int(time.time()))
subG=nx.k_core(G,1)
# nx.draw_networkx(G,with_labels=True)
# plt.show()
print("This graph has", nx.number_of_nodes(subG), "nodes and ", nx.number_of_edges(subG),"edges.")
print("The nodes of G are:")
print(nx.nodes(subG))
print("The edges of G are:")
print(nx.edges(subG,None))
print("The core numbers of G are:")
print(nx.core_number(subG))

print("The optimal 2 dimensional structural information and core based partition of G are:")
print(td.optimalTwoDimensionalStructuralInformation(subG))
# print("All of the partitions are：")
# for idx, item in enumerate(td.partition(list(nx.nodes(subG)))):
#     print(item)

print("The 2 dimensional structural information and corresponding partition of G are:")
print(td.twoDimensionalStructuralInformation(subG))
    
adjm=nx.to_numpy_matrix(subG)
numpy.savetxt('matrix.csv', adjm, delimiter = ',') 
# adjmatrix=readAdjMatrix("m")
# G=nx.from_numpy_matrix(adjmatrix,False,None)
# nx.draw(G)
# plt.show()
# print("The optimal 2 dimensional structural information and core based partition of G are:")
# print(td.optimalTwoDimensionalStructuralInformation(G))

# print("The 2 dimensional structural information and corresponding partition of G are:")
# print(td.twoDimensionalStructuralInformation(G))

# adjmatrix=readAdjMatrix("./data/matrix")
# for i in range(len(adjmatrix)):
#     for j in range(i, len(adjmatrix)):
#         if adjmatrix[i,j]==1:
#             print(i+1, j+1, 1)
