import numpy as np
from dijkstra.dijkstra import dijkstra, shortest_path
import matplotlib.pyplot as plt
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

        #History used for plotting
        self.path_history = []
        self.explored_nodes_history = []
        self.points_chosen_for_training_history = []

    def fill_graph_gridwise(self,limits,density_vector,environment):
        """In this function the graph is meshed with a grid, and each neighbor of a node
        is its extension along one dimension."""

        #Since we can only deal with integers in this dictionary, let's multiply everything and assign scaling factor
        scaling_factors = [int(1./density) for density in density_vector]
        self.scaling_factors = scaling_factors

        grid_ticks = [] #Contains all coordinates in each dimension
        grid_spacings = [] #Spacing along each dimension
        for d in range(self.dimension):
            s = scaling_factors[d]
            #These are all the components of our axis
            if d==0:
                grid_ticks = [np.mgrid[limits[d][0]*s:(limits[d][1]+density_vector[d])*s:density_vector[d]*s]]
            else:
                grid_ticks = [*grid_ticks, np.mgrid[limits[d][0]*s:(limits[d][1]+density_vector[d])*s:density_vector[d]*s]]


            grid_spacings += [abs(grid_ticks[d][1] - grid_ticks[d][0])]


        #Create all the nodes of the graph by meshing according to the coordinates
        all_nodes = np.array(np.meshgrid(*grid_ticks)).T.reshape([-1,self.dimension])

        print('The graph contains ',len(all_nodes), 'nodes.')
        #embed()

        #Adding the nodes and their neighbors to the graph dictionary
        for node in all_nodes:
            if environment._in_obst(self.transform_from_grid(node)) == False:
                self.g[tuple(node)] = self.find_neighbor_nodes_and_assign_cost(node,environment,grid_spacings)

        self.clip_to_grid(environment.start_state)
        self.clip_to_grid(environment.goal_state)
        #embed()
        return

    def find_neighbor_nodes_and_assign_cost(self,node,environment,grid_spacings):
        """We have a feasible state, and we would like to create a dictionary with all its
        feasible neighbors and the cost to go to each of them."""
        neighbors_and_costs = {}
        node = np.rint(node)
        for d in range(self.dimension):

            node = np.rint(node) #Only way I found to get rid of the float bug.

            #This is how much we want to increment our state to reach a neighbor state
            displacement_vector = np.zeros(self.dimension)
            displacement_vector[d] = grid_spacings[d]

            positive_neighbor = node + displacement_vector
            negative_neighbor = node - displacement_vector

            #embed()

            #If the positive neighbor is not in collision we suppose that it is reachable.
            if environment._in_obst(self.transform_from_grid(positive_neighbor)) == False:
                #Let's add the node in the graph to avoid possible bugs. If it has not been added yet
                #it is okay, since it will be updated with its own neighbors.
                try:
                    self.g[tuple(positive_neighbor)]
                except:
                    #embed()
                    self.g[tuple(positive_neighbor)] = {} #We don't want to erase already existing information
                if environment.system.is_feasible_LQR_path(self.transform_from_grid(node), self.transform_from_grid(positive_neighbor),environment):
                    neighbors_and_costs[tuple(positive_neighbor)] = environment.system.solve_optimal_control_cost(self.transform_from_grid(node),self.transform_from_grid(positive_neighbor))

            # We do the same for the negative neighbor
            if environment._in_obst(self.transform_from_grid(negative_neighbor)) == False:
                try:
                    self.g[tuple(negative_neighbor)]
                except:
                    self.g[tuple(negative_neighbor)] = {} #We don't want to erase already existing information
                if environment.system.is_feasible_LQR_path(node,self.transform_from_grid(negative_neighbor),environment):
                    neighbors_and_costs[tuple(negative_neighbor)] = environment.system.solve_optimal_control_cost(node,self.transform_from_grid(negative_neighbor))

        return neighbors_and_costs

    def clip_to_grid(self,node):
        """Approximates the node as the nearest node by assigning it to a grid element
        Since the integer update this should do nothing and return automatically"""
        node_in_grid = self.transform_to_grid(node)
        try:
            self.g[tuple(node_in_grid)]
            return
        except:
            nearest_neighbor = self.transform_to_grid(np.array([4,0,4,0]))
            self.g[tuple(node)] = {tuple(nearest_neighbor):0}
            self.g[tuple(nearest_neighbor)] = {tuple(node): 0}
        return

    def transform_to_grid(self,node):
        node_in_grid = np.zeros(self.dimension)
        for d in range(self.dimension):
            node_in_grid[d] = int(node[d]*self.scaling_factors[d])
        return node_in_grid

    def transform_from_grid(self,node_in_grid):
        node = np.zeros(self.dimension)
        for d in range(self.dimension):
            node[d] = float(node_in_grid[d])/float(self.scaling_factors[d]) #Scaling factors should never be zero so ok
        return node


    def get_shortest_path(self,start_state,end_state):
        start_state = self.transform_to_grid(start_state)
        end_state = self.transform_to_grid(end_state)
        path_in_grid = shortest_path(self.g,tuple(start_state),tuple(end_state))
        shortest_path_off_grid = []
        for node in path_in_grid:
            node_off_grid = self.transform_from_grid(node)
            shortest_path_off_grid += [node_off_grid]
        return shortest_path_off_grid

    def delete_edge(self,node_start,node_end):
        node_start = self.transform_to_grid(node_start)
        node_end = self.transform_to_grid(node_end)
        del self.g[tuple(node_start)][tuple(node_end)]
        return

    def delete_edge_t(self,node_start,node_end):
        del self.g[tuple(node_start)][tuple(node_end)]
        return

    def assign_to_graph(self,node_state,next_state,cost):
        node_state = self.transform_to_grid(node_state)
        next_state = self.transform_to_grid(next_state)
        self.g[tuple(node_state)][tuple(next_state)] = cost
        return

    def plot_graph_history(self,environment):
        """Creating as much plots as iterations numbered accordingly"""
        for i in range(len(self.explored_nodes_history)):

            # Retrieving data
            path_history = self.path_history[i]
            explored_nodes_history = self.explored_nodes_history[i]
            point_chosen_for_training = self.points_chosen_for_training_history[i]


            # ---------------------- PLOTTING MAP --------------------------------------- #
            ax = plt.gca()
            for xo, yo, ro in zip(environment.obst_X, environment.obst_Y, environment.obst_R):
                c = plt.Circle((xo, yo), ro, color='black', alpha=1.0)
                ax.add_artist(c)

            r = plt.Rectangle((environment.xg_lower, environment.yg_lower), environment.xg_upper - environment.xg_lower, environment.yg_upper - environment.yg_lower,
                              color='g', alpha=0.3, hatch='/')
            ax.add_artist(r)
            # --------------------------------------------------------------------------- #

            # ------------------ PLOTTING PATH HISTORY ---------------------------------- #
            list_of_xs = [state[0] for state in path_history]
            list_of_ys = [state[2] for state in path_history]

            plt.plot(list_of_xs,list_of_ys,color="green")
            # --------------------------------------------------------------------------- #


            # ----------------- PLOTTING EXPLORED NODES IN RED -------------------------- #
            for explored_node in explored_nodes_history:
                explored_node = self.transform_from_grid(explored_node)
                explored_x = explored_node[0]
                explored_y = explored_node[2]
                plt.plot([explored_x], [explored_y], marker='o', linewidth=2, color='r', markersize=5)
            # --------------------------------------------------------------------------- #

            # ------------ PLOTTING NEXT POINT TO TRAIN ON IN BLUE ---------------------- #
            pcft_x = point_chosen_for_training[0]
            pcft_y = point_chosen_for_training[2]
            plt.plot([pcft_x], [pcft_y], marker='o', linewidth=2, color='blue', markersize=8)
            # --------------------------------------------------------------------------- #


            plt.xlim([environment.x_lower, environment.x_upper])
            plt.ylim([environment.y_lower, environment.y_upper])

            #Saves the figure
            plt.savefig("plots/"+str(i)+".png")

            #Clears the figure to create another one
            plt.clf()

        return


