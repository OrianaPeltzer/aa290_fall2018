import numpy as np
from dijkstra.dijkstra import dijkstra, shortest_path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import os
from IPython import embed
import copy
from math import log

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

        #FMT-only attributes
        self.fmt_total_g = {} # This graph will be used to store all the edges we know about. It saves a little computation time.
        self.forbidden_edges = {} #This is all the edges that fmt must not consider
        self.samples = [] #Contains the full list of all the samples
        self.node_by_node_optimal_paths_to_go = {}

        #Explored means that ppo has (successfully or not) trained with this node
        self.explored_nodes = []
        self.reward_history = []

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
                if environment.system.is_feasible_LQR_path(self.transform_from_grid(node),self.transform_from_grid(negative_neighbor),environment):
                    neighbors_and_costs[tuple(negative_neighbor)] = environment.system.solve_optimal_control_cost(self.transform_from_grid(node),self.transform_from_grid(negative_neighbor))

        return neighbors_and_costs

    def fill_graph_fmtwise(self,limits,density_vector,environment,k0=300):
        """In this function a grid is created and samples are uniformly taken from it. FMT is applied to fill in the
        graph optimally.
        We only use this function the first time we sample from the grid. So we are sure that there aren't any explored
        nodes that we want to consider as goal state as well."""


        # -------------------- STEP 1: CREATING THE GRID ------------------------------------------------------------

        # Since we can only deal with integers in this dictionary, let's multiply everything and assign scaling factor
        scaling_factors = [1. / density for density in density_vector]
        self.scaling_factors = scaling_factors
        grid_ticks = []  # Contains all coordinates in each dimension
        grid_spacings = []  # Spacing along each dimension
        for d in range(self.dimension):
            s = scaling_factors[d]
            # These are all the components of our axis
            if d == 0:
                grid_ticks = [np.mgrid[int(limits[d][0] * s): int((limits[d][1] + density_vector[d]) * s):int(density_vector[d] * s)]]
            else:
                grid_ticks = [*grid_ticks,
                              np.mgrid[int(limits[d][0] * s):int((limits[d][1] + density_vector[d]) * s):int(density_vector[d] * s)]]

            grid_spacings += [abs(grid_ticks[d][1] - grid_ticks[d][0])]

        # Create all the nodes of the graph by meshing according to the coordinates
        all_nodes = np.array(np.meshgrid(*grid_ticks)).T.reshape([-1, self.dimension])
        #embed()
        # -----------------------------------------------------------------------------------------------------------


        # --------------------------- STEP 2: SAMPLE FROM GRID ------------------------------------------------------
        print("The grid contains ", len(all_nodes), "nodes. Let's randomly sample from it to create our graph")
        #num_nodes_to_sample = int(0.01*float(len(all_nodes)))
        num_nodes_to_sample = 25000
        print("We will sample ", num_nodes_to_sample, "nodes from the grid.")

        data = copy.deepcopy(all_nodes).tolist()

        samples = []

        while len(data) > len(all_nodes)-num_nodes_to_sample:
            index = np.random.randint(len(data))
            elem = data[index]
            # direct deletion, no search needed
            del data[index]
            if environment._in_obst(self.transform_from_grid(elem)) == False:
                samples += [tuple(elem)]
        # -----------------------------------------------------------------------------------------------------------

        # Make sure that start and goal are in samples
        if tuple(self.transform_to_grid(environment.start_state)) not in samples:
            samples += [tuple(self.transform_to_grid(environment.start_state))]
        if tuple(self.transform_to_grid(environment.goal_state)) not in samples:
            samples += [tuple(self.transform_to_grid(environment.goal_state))]

        self.samples = samples

        self.kd_tree = cKDTree(np.array(samples))

        # ------------------------------------------------------------------------
        start = tuple(self.transform_to_grid(environment.start_state))
        goal = tuple(self.transform_to_grid(environment.goal_state))

        print("Nodes sampled. Now looking for best path")

        self.node_by_node_optimal_paths_to_go = {tuple(self.transform_to_grid(environment.start_state)): []}
        self.apply_fmt_for_shortest_path(start,goal,environment,k0)


        return

    def apply_fmt_for_shortest_path(self,start,goal,environment,k0):
        """Applies fmt to find shortest path from start to defined goals, avoiding forbidden links.
        start and goal, unlike a lot of other functions here, must be in graph form already!"""

        print("Starting an iteration of FMT")

        # Now to create the links, we start from start_state and perform FMT iterations.
        # The goal is only the goal at first. In another function we will fill in self.g with the shortest
        # paths to all explored states
        V_closed = []
        V_unvisited = copy.deepcopy(self.samples) #Unnecessary?
        V = copy.deepcopy(self.samples)
        V_open = [start]

        node_by_node_costs = {}

        #Delete the start state from V_unvisited since we put it in V_open
        V_unvisited.remove(start)

        k = 35
        goal_state = goal
        z = start


        print("Entering while")

        iteration = 0
        terminated = False
        while terminated == False:
            Nz = self.get_samples_near_FAST(z, k)
            V_open_new = []
            X_near = []

            #embed()
            for elt in V_unvisited:
                if elt in Nz:
                    X_near += [elt]

            for x in X_near:

                # Here we try to avoid an order of magnitude of complexity by taking advantage of already stored values.
                pre_computed_path = False
                #try:
                #    path_to_x = self.node_by_node_optimal_paths_to_go[x]
                #    path_forbidden = False
                #    #We must check if there is no forbidden edge in the path
                #    cost_to_x = 0
                #    for kx in range(1,len(path_to_x)):
                #        n1 = tuple(path_to_x[kx-1])
                #        n2 = tuple(path_to_x[kx])
                #        try:
                #            self.forbidden_edges[n1][n2]
                #            path_forbidden = True
                #        except:
                #            try:
                #                little_ctx = self.g[n1][n2] #This way if it has been trained on we have the good value
                #                cost_to_x += little_ctx
                #            except:
                #                cost_to_x += environment.system.solve_optimal_control_cost(self.transform_from_grid(n1),self.transform_from_grid(n2))
                #    if not path_forbidden:
                #        best_y = self.node_by_node_optimal_paths_to_go[x][-1]
                #       V_open_new += [x]
                #        V_unvisited.remove(x)
                #        pre_computed_path = True
                #        #The path being feasible we can update the optimal cost of
                #        node_by_node_costs[x] = cost_to_x
                #    else:
                #        del self.node_by_node_optimal_paths_to_go[x]
                #except:
                #    pass

                if pre_computed_path == False: #Now we really scan all the points

                    #embed()
                    Nx = self.get_samples_near_FAST(x,k)
                    #embed()

                    # Y_near = inter(Nx,V_open). At the first iteration this should only be the starting point
                    best_y = None #This would stay the case if there were no y
                    min_cost = np.inf  # This is the cost we will try to minimize
                    for y in Nx:
                        # ----- Forbidden edge test ----- #
                        forbidden = False #Here we eliminate the case where edges are forbidden. But we still explore unforbidden
                                          # edges with a chance to find an optimal neighbor one, which would be valuable
                        try:
                            self.forbidden_edges[y][x]
                            forbidden = True # If this works it means y -> x is forbidden so both y and x have been trained on.
                        except:
                            pass
                        # ------------------------------- #

                        if y in V_open and not forbidden: #So this point is in Y_near theoretically

                            #Get the optimal cost to go at nx
                            try:
                                cost_of_y = node_by_node_costs[y]
                                len_y = len(self.node_by_node_optimal_paths_to_go[y])
                            except:
                                cost_of_y = 0. #This should mean that nx is the starting point
                                len_y = 0
                                assert(y == start)

                            #Now try to get the cost to go from y to x
                            try:
                                cost_to_go = self.fmt_total_g[y][x]
                            except:
                                cost_to_go = environment.system.solve_optimal_control_cost(self.transform_from_grid(y),self.transform_from_grid(x))
                                try:
                                    self.fmt_total_g[y][x] = cost_to_go #This means that we already had the optimal cost to go from y to somewhere else
                                except:
                                    self.fmt_total_g[y] = {x: cost_to_go}
                            #Now we can find the total cost of going to x passing by y.

                            cost_to_x_by_y = (cost_of_y*len_y + cost_to_go)/(1+len_y) #DP equation
                            if cost_to_x_by_y <= min_cost:
                                best_y = y #This is the temporary min
                                min_cost = cost_to_x_by_y
                    # Now we have y = argmin(c(y) + c(y,x)).
                    # Here we make the very lazy assumption that if two nodes are neighbors, their path is collision free
                    # if and only if their intersection point is collision free. This is approximately true from the fact that here
                    # All our obstacles are circles and moreover the path that a controller would choose to take may or may not
                    # traverse obstacles, so we count on the small euclidean distance between neighbors to make this work.

                    if best_y != None:
                        #Middle_node = 0.5*(np.array(self.transform_from_grid(best_y))+np.array(self.transform_from_grid(x)))
                        #if environment._in_obst(Middle_node) == False: #If path is collision free (see above)
                        #    V_open_new += [x]
                        #    #embed()
                        #    V_unvisited.remove(x)
                        #    node_by_node_costs[x] = copy.deepcopy(min_cost)  # Update the dictionary of optimal costs
                        #    self.node_by_node_optimal_paths_to_go[x] = copy.deepcopy(self.node_by_node_optimal_paths_to_go[best_y])
                        #    self.node_by_node_optimal_paths_to_go[x] += [best_y]
                        V_open_new += [x]
                        V_unvisited.remove(x)
                        node_by_node_costs[x] = copy.deepcopy(min_cost)  # Update the dictionary of optimal costs
                        self.node_by_node_optimal_paths_to_go[x] = copy.deepcopy(
                            self.node_by_node_optimal_paths_to_go[best_y])
                        self.node_by_node_optimal_paths_to_go[x] += [best_y]


            # --------- Here ends both loops, we are now just in the while and have iterated on all X_near -------- #

                # Performs V_open = union(V_open,V_open_new)\z
            for x_new in V_open_new:
                if x_new not in V_open: #This eliminates the case where we would mistakenly add several of the same node
                    V_open += [x_new]
            try:
                V_open.remove(z) #If z is inside it will delete it, since we know it is only there once
            except:
                pass

            V_closed += [z]

            if len(V_open) == 0:
                print("\n FMT failed for iteration with goal state ", goal)
                #embed()
                return False # Failure :'( or there is a bug or discretization is not good enough or we are in the
                             # special cases of there being forbidden and explored edges in which case this is
                             # normal and interesting at the same time

            if np.linalg.norm(np.subtract(z,goal_state)) <= 0.05:
                terminated = True

            min_cost2 = np.inf #This has to change since we know V_open is not null
            for y2 in V_open:
                if y2 == z:
                    print("Error! Plenty of zs!!")
                cy = node_by_node_costs[y2] #If you reach out to that it means that if its old value is
                                                 # obsolete due to a forbidden edge, then it has been updated.
                if cy < min_cost2:
                    min_cost2 = cy
                    z = y2


            iteration += 1
            if iteration % 1000 == 0:
                print("Iteration ", iteration, " done. Explored state percentage: ")
                print((1.-float(len(V_unvisited))/float(len(self.samples)))*100.)
                #print(len(V_unvisited))





        print("\n FMT succeeded in finding a path with goal state being ", goal)
        #Now we are done, and if the code arrives here it means that z = goal. So let's add the elementary costs
        #to the graph self.g using the path we found here.
        path = self.node_by_node_optimal_paths_to_go[goal_state] #goal state has been transformed to the grid so ok
        path += [goal_state]
        for k in range(1,len(path)):
            node = path[k]
            previous_node = path[k-1]
            cost_to_go = environment.system.solve_optimal_control_cost(self.transform_from_grid(previous_node),self.transform_from_grid(node))
            try:
                self.g[tuple(previous_node)][tuple(node)] = copy.deepcopy(cost_to_go) #if previous node is already there
            except:
                self.g[tuple(previous_node)] = {tuple(node): copy.deepcopy(cost_to_go)}
        return

    def get_samples_near_FAST(self, z, k):
        """Gets the k closest samples to z in terms of norm.
        The cost in between two samples is not the norm but the norm-approximation should
        work fine here.
        Samples is a list of tuples of elements in grid form.
        We would need to rescale the elements if grid form was not uniform along all dimensions.
        For the rotation we will only use examples where it is and not rescale.

        Uses an efficient KD-Tree query to lookup nearest neighbors."""
        dists, idxs = self.kd_tree.query(z, k=k+1)
        return [self.samples[i] for i in idxs[1:]]

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
        """start and end are in real world coordinates and need to be transformed to grid.
        Dijkstra is used in this function."""
        start_state = self.transform_to_grid(start_state)
        end_state = self.transform_to_grid(end_state)
        path_in_grid = shortest_path(self.g,tuple(start_state),tuple(end_state))
        shortest_path_off_grid = []
        for node in path_in_grid:
            node_off_grid = self.transform_from_grid(node)
            shortest_path_off_grid += [node_off_grid]
        return shortest_path_off_grid, path_in_grid

    def delete_edge(self,node_start,node_end):
        node_start = self.transform_to_grid(node_start)
        node_end = self.transform_to_grid(node_end)
        del self.g[tuple(node_start)][tuple(node_end)]
        return

    def delete_edge_in_grid(self,node_start,node_end):
        del self.g[tuple(node_start)][tuple(node_end)]
        return

    def delete_edge_fmt_total(self,n1,n2):
        node1 = self.transform_to_grid(n1)
        node2 = self.transform_to_grid(n2)
        try:
            del self.fmt_total_g[tuple(node1)][tuple(node2)]
        except:
            pass
        return

    def delete_edge_fmt_total_in_grid(self,n1,n2):
        try:
            del self.fmt_total_g[tuple(n1)][tuple(n2)]
        except:
            pass
        return

    def assign_to_graph(self,node_state,next_state,cost):
        node_state = self.transform_to_grid(node_state)
        next_state = self.transform_to_grid(next_state)
        self.g[tuple(node_state)][tuple(next_state)] = cost
        self.fmt_total_g[tuple(node_state)][tuple(next_state)] = cost
        return


    def assign_to_graph_in_grid(self,node_state,next_state,cost):
        self.g[tuple(node_state)][tuple(next_state)] = cost
        self.fmt_total_g[tuple(node_state)][tuple(next_state)] = cost
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
            #for xo, yo, ro in zip(environment.obst_X, environment.obst_Y, environment.obst_R):
            #    c = plt.Circle((xo, yo), ro, color='black', alpha=1.0)
            #    ax.add_artist(c)

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


    def plot_graph_last(self,environment,iteration,name="Test1"):
        """Creating as much plots as iterations numbered accordingly"""
        i = iteration

        # Retrieving data
        path_history = self.path_history[i]
        explored_nodes_history = self.explored_nodes_history[i]
        point_chosen_for_training = self.points_chosen_for_training_history[i]


        # ---------------------- PLOTTING MAP --------------------------------------- #
        ax = plt.gca()
        #for xo, yo, ro in zip(environment.obst_X, environment.obst_Y, environment.obst_R):
        #    c = plt.Circle((xo, yo), ro, color='black', alpha=1.0)
        #    ax.add_artist(c)

        r = plt.Rectangle((environment.xg_lower, environment.yg_lower), environment.xg_upper - environment.xg_lower, environment.yg_upper - environment.yg_lower,
                          color='g', alpha=0.3, hatch='/')
        ax.add_artist(r)
        # --------------------------------------------------------------------------- #

        # ------------------ PLOTTING PATH HISTORY ---------------------------------- #
        list_of_xs = [state[0] for state in path_history]
        list_of_ys = [state[1] for state in path_history]

        plt.plot(list_of_xs,list_of_ys,color="green")
        # --------------------------------------------------------------------------- #


        # ----------------- PLOTTING EXPLORED NODES IN RED -------------------------- #
        for explored_node in explored_nodes_history:
            explored_node = self.transform_from_grid(explored_node)
            explored_x = explored_node[0]
            explored_y = explored_node[1]
            plt.plot([explored_x], [explored_y], marker='o', linewidth=2, color='r', markersize=5)
        # --------------------------------------------------------------------------- #

        # ----------------- PLOTTING LAST EXPLORED NODE IN PURPLE ------------------- #
        explored_node = explored_nodes_history[-1]
        explored_node = self.transform_from_grid(explored_node)
        explored_x = explored_node[0]
        explored_y = explored_node[1]
        plt.plot([explored_x], [explored_y], marker='o', linewidth=2, color='tab:purple', markersize=5)
        # --------------------------------------------------------------------------- #

        # ------------ PLOTTING NEXT POINT TO TRAIN ON IN BLUE ---------------------- #
        pcft_x = point_chosen_for_training[0]
        pcft_y = point_chosen_for_training[1]
        plt.plot([pcft_x], [pcft_y], marker='o', linewidth=2, color='blue', markersize=8)
        # --------------------------------------------------------------------------- #

        # Plotting training result
        plt.text(-2.,-2.,str(int(self.reward_history[i])),color="k")

        plt.xlim([environment.x_lower, environment.x_upper])
        plt.ylim([environment.y_lower, environment.y_upper])

        #Saves the figure
        if not os.path.exists("plots/"+name):
            os.makedirs("plots/"+name)
        plt.savefig("plots/"+name+"/"+str(i)+".png")

        #Clears the figure to create another one
        plt.clf()

        return




