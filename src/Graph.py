import numpy as np
from dijkstra.dijkstra import dijkstra, shortest_path
from IPython import embed

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

        #Explored means that ppo has successfully trained with this node
        self.explored_nodes = []

    def fill_graph_gridwise(self,limits,density_vector,environment):
        """In this function the graph is meshed with a grid, and each neighbor of a node
        is its extension along one dimension."""
        grid_ticks = [] #Contains all coordinates in each dimension
        grid_spacings = [] #Spacing along each dimension
        for d in range(self.dimension):
            #These are all the components of our axis
            if d==0:
                grid_ticks = [np.mgrid[limits[d][0]:limits[d][1]:density_vector[d]]]
            else:
                grid_ticks = [*grid_ticks, np.mgrid[limits[d][0]:limits[d][1]:density_vector[d]] ]


            grid_spacings += [abs(grid_ticks[d][1] - grid_ticks[d][0])]


        #Create all the nodes of the graph by meshing according to the coordinates
        all_nodes = np.array(np.meshgrid(*grid_ticks)).T.reshape([-1,self.dimension])

        print('The graph contains ',len(all_nodes), 'nodes.')
        #embed()

        #Adding the nodes and their neighbors to the graph dictionary
        for node in all_nodes:
            if environment._in_obst(node) == False:
                self.g[tuple(node)] = self.find_neighbor_nodes_and_assign_cost(node,environment,grid_spacings)

    def find_neighbor_nodes_and_assign_cost(self,node,environment,grid_spacings):
        """We have a feasible state, and we would like to create a dictionary with all its
        feasible neighbors and the cost to go to each of them."""
        neighbors_and_costs = {}
        displacement_vector = np.zeros(self.dimension)
        for d in range(self.dimension):
            #This is how much we want to increment our state to reach a neighbor state
            displacement_vector[d] = grid_spacings[d]

            positive_neighbor = node + displacement_vector
            negative_neighbor = node - displacement_vector

            #If the positive neighbor is not in collision we suppose that it is reachable.
            if environment._in_obst(positive_neighbor) == False:
                #Let's add the node in the graph to avoid possible bugs. If it has not been added yet
                #it is okay, since it will be updated with its own neighbors.
                try:
                    self.g[tuple(positive_neighbor)]
                except:
                    #embed()
                    self.g[tuple(positive_neighbor)] = {} #We don't want to erase already existing information
                if environment.system.is_feasible_LQR_path(node, positive_neighbor,environment):
                    neighbors_and_costs[tuple(positive_neighbor)] = environment.system.solve_optimal_control_cost(node,positive_neighbor)

            # We do the same for the negative neighbor
            if environment._in_obst(negative_neighbor) == False:
                try:
                    self.g[tuple(negative_neighbor)]
                except:
                    self.g[tuple(negative_neighbor)] = {} #We don't want to erase already existing information
                if environment.system.is_feasible_LQR_path(node,negative_neighbor,environment):
                    neighbors_and_costs[tuple(negative_neighbor)] = environment.system.solve_optimal_control_cost(node,negative_neighbor)

        return neighbors_and_costs

    def get_shortest_path(self,start_state,end_state):
        return shortest_path(self.g,start_state,end_state)

    def delete_edge(self,node_start,node_end):
        del self.g[node_start][node_end]
        return


