from utils import uniform, sample
from Graph import Graph
from numpy.random import uniform
import numpy as np
import copy
from IPython import embed

def random(starts, N_new, problem, **kwargs):
    if len(starts) == 0:
        print('empty starts list!', flush=True)
        return list()
    
    num_random_steps = 10 if 'num_random_steps' not in kwargs else kwargs['num_random_steps']
    total_runs = 1000 if 'total_runs' not in kwargs else kwargs['total_runs']
    
    while len(starts) < total_runs:
        start = sample(starts)
        problem.reset_to_state(start)
        for t in range(num_random_steps):
            action_t = uniform(low=problem.action_space.low,
                               high=problem.action_space.high)

            state_t, _, _, _ = problem.step(action_t, ret_state=True)

            if (problem.env_name == 'PlanarQuad-v0' 
                and problem.env.unwrapped._in_obst(state_t)):
                    break

            starts.append(state_t)

    new_starts = sample(starts, size=N_new)
    return new_starts


def update_backward_reachable_set(starts, **kwargs):
    br_engine = kwargs['br_engine'];
    problem = kwargs['problem']
    variation = kwargs['variation']

    # (1) Check if start is in backreach set (if so, stop expanding).
    if variation == 1 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
        print('Variation 1 condition curriculum.py!', flush=True)
        return

    br_engine.update_and_compute_backward_reachable_set(starts, 
                                                        plot=kwargs['debug'],
                                                        curr_train_iter=kwargs['curr_train_iter']);


def sample_from_backward_reachable_set(N_new, **kwargs):
    br_engine = kwargs['br_engine'];
    problem = kwargs['problem']
    variation = kwargs['variation']

    # (2) When you do reach it, only sample the start
    if variation == 2 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
        print('Variation 2 condition curriculum.py!', flush=True)
        return [problem.env.unwrapped.start_state]

    new_starts_arr = br_engine.sample_from_grid(size=N_new, method=kwargs['brs_sample']);
    
    return [new_starts_arr[i] for i in range(new_starts_arr.shape[0])];


def backward_reachable(starts, N_new, problem, **kwargs):
    update_backward_reachable_set(starts, **locals());
    return sample_from_backward_reachable_set(N_new, **locals());


def graph(state_that_has_just_been_trained_on, start_state, goal_state, my_graph, pct_successful,ep_mean_rews,k0=0,**kwargs):

    success_score = pct_successful*1000
    cost_to_go = np.mean(ep_mean_rews)-success_score
    real_cost = -((1.0-pct_successful)*cost_to_go) #Cost once ppo has trained on this node

    # Add edge to graph linking goal state and trained node
    my_graph.assign_to_graph(state_that_has_just_been_trained_on,goal_state,real_cost)
    #my_graph.g[state_that_has_just_been_trained_on][goal_state] = real_cost


    #Tell the graph that this node is explored
    my_graph.explored_nodes += [my_graph.transform_to_grid(state_that_has_just_been_trained_on)]

    next_path = my_graph.get_shortest_path(start_state,goal_state) #This outputs a list of tuples

    if len(next_path) <= 2:
        return True, my_graph

    np2 = my_graph.transform_to_grid(next_path[-2])
    explored_nodes = my_graph.explored_nodes

    #embed()

    if np.isin(explored_nodes,np2).all(axis=1).any():
        print("I have already explored node ", np2, "so I will delete edge linking it to ",next_path[-3])
        my_graph.delete_edge(next_path[-3],next_path[-2])
        next_point_to_train_on = list(next_path[-3]) #Need to convert tuples to lists
    else:
        print("Node ", np2, "is new! Let's delete its link to the goal and explore it")
        my_graph.delete_edge(next_path[-2], goal_state)
        next_point_to_train_on = list(next_path[-2])

    print("Next path found. It is length ", len(next_path), ":")
    print(next_path)

    #Let's save the new graoh as we intend to train it
    my_graph.path_history += [next_path]
    my_graph.explored_nodes_history += [copy.deepcopy(explored_nodes)]
    my_graph.points_chosen_for_training_history += [next_point_to_train_on]

    # return a new starting point in real coordinates
    return next_point_to_train_on, my_graph


def graph_fmt(state_that_has_just_been_trained_on, start_state, goal_state, my_graph, pct_successful,ep_mean_rews,environment,k0=300,**kwargs):
    gamma=0.5
    M = 1000 #Very big (failure)

    cost_to_go = float(environment.control_cost)

    print("Percentage successful: ", str(pct_successful))

    success_after_training = float(ep_mean_rews[-1])/2000. + 0.5 #Map the reward to a success score btw -1 and 1
    if pct_successful <=0.1:
        real_cost = M
    else:
        real_cost = cost_to_go*pct_successful/(2*(gamma-1)) + cost_to_go*(gamma-2)/(2*(gamma-1)) #Cost once ppo has trained on this node

    my_graph.reward_history += [ep_mean_rews[-1]]

    # Add edge to graph linking goal state and trained node
    my_graph.assign_to_graph(state_that_has_just_been_trained_on,goal_state,real_cost) #This also updates/overwrites total_fmt costs.
    my_graph.fmt_total_g[tuple(my_graph.transform_to_grid(state_that_has_just_been_trained_on))][tuple(my_graph.transform_to_grid(goal_state))] = real_cost

    #Tell the graph that this node is explored
    my_graph.explored_nodes += [my_graph.transform_to_grid(state_that_has_just_been_trained_on)]

    #In all cases, we need to re-apply fmt for the goal because we changed some costs. We do this before selecting next point
    try:
        del my_graph.node_by_node_optimal_paths_to_go[tuple(my_graph.transform_to_grid(goal_state))]
    except:
        pass

    #Now let's apply fmt again to see if we could add another path leading to the goal coming from somewhere else
    my_graph.apply_fmt_for_shortest_path(tuple(my_graph.transform_to_grid(start_state)), tuple(my_graph.transform_to_grid(goal_state)), environment, k0)


    next_path = my_graph.get_shortest_path(start_state,goal_state) #This outputs a list of tuples

    if len(next_path) <= 2:
        return True, my_graph

    np2 = my_graph.transform_to_grid(next_path[-2])
    np3 = my_graph.transform_to_grid(next_path[-3])
    explored_nodes = my_graph.explored_nodes

    #embed()

    if np.isin(explored_nodes,np2).all(axis=1).any():
        print("I have already explored node ", np2, "so I will delete edge linking it to ",next_path[-3])

        # The delete operation is more complex in the case of fmt since we also forbid the edge and apply fmt to the node
        # that we just abandoned
        my_graph.delete_edge(next_path[-3],next_path[-2])
        my_graph.delete_edge_fmt_total(next_path[-3],next_path[-2])
        try:
            my_graph.forbidden_edges[tuple(np3)][tuple(np2)] = True
        except:
            my_graph.forbidden_edges[tuple(np3)] = {tuple(np2): True}
        try:
            del my_graph.node_by_node_optimal_paths_to_go[np2]
        except:
            pass

        # Now we apply fmt to find a path to np2, the node that we just abandoned, and put it in self.g
        my_graph.apply_fmt_for_shortest_path(tuple(my_graph.transform_to_grid(start_state)),tuple(np2),environment,k0)

        next_point_to_train_on = list(next_path[-3]) #Need to convert tuples to lists
    else:
        print("Node ", np2, "is new! Let's delete its link to the goal and explore it")
        my_graph.delete_edge(next_path[-2], goal_state) #delete from graph.g, the graph to search from
        my_graph.delete_edge_fmt_total(next_path[-2], goal_state) #Also delete from stored values in big storage graph
        next_point_to_train_on = list(next_path[-2])


    print("Next path found. It is length ", len(next_path), ":")
    print(next_path)

    #Let's save the new graoh as we intend to train it
    my_graph.path_history += [next_path]
    my_graph.explored_nodes_history += [copy.deepcopy(explored_nodes)]
    my_graph.points_chosen_for_training_history += [next_point_to_train_on]

    # return a new starting point in real coordinates
    return next_point_to_train_on, my_graph