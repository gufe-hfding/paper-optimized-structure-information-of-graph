import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import scipy

import readAdjMatrix
import twoDimensionalStructuralInformation as td
from readAdjMatrix import readAdjMatrix

numpy.random.seed(int(time.time()))
n=numpy.random.randint(4,8)
p=numpy.random.uniform(0.3,0.7)


adjm=numpy.loadtxt(open("matrix.csv","rb"),delimiter=",",skiprows=0) 
G=nx.from_numpy_matrix(adjm)


adjmatrix=readAdjMatrix("./data/matrix")
G=nx.from_numpy_matrix(adjmatrix,False,None)
part=[[0,1],[2,3],[6,7,8],[9,10,11],[12,13,14,15],[19,20,21,22],[23,24,25,26,27,28],[5],[6],[17],[18],[19]]
print(td.structuralInformation(G,part))
print(nx.core_number(G))
