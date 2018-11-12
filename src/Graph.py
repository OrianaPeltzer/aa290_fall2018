import numpy as np

class Graph():
    """Graph is an object that can be used both by dijkstra and in the curriculum.
    The attribute self.g is a dictionary structured as such:
    self.g = { node1: {neighbor_node1: cost_to_go, neighbor_node2: cost_to_go},
               node2:  ... }"""

    def __init__(self, dimension):
        #This is the dimension of each element of the graph
        self.dimension = dimension
        #This is an empty graph
        self.g = {}


    def add_node(self,node):
        self.g[node] = {}

    def fill_graph_gridwise(self,limits,density_vector):
        """In this function the graph is meshed with a grid, and each neighbor of a node
        is its extension along one dimension."""
        for d in self.dimension:
            
