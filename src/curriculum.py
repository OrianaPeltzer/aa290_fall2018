from utils import uniform, sample
from Graph import Graph
from numpy.random import uniform
import numpy as np
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


def graph(state_that_has_just_been_trained_on, start_state, goal_state, my_graph, pct_successful,ep_mean_rews,**kwargs):

    real_cost = ((1.0-pct_successful)**2)*np.mean(ep_mean_rews)/1000. #Cost once ppo has trained on this node

    # Add edge to graph linking goal state and trained node
    my_graph.assign_to_graph(state_that_has_just_been_trained_on,goal_state,real_cost)
    #my_graph.g[state_that_has_just_been_trained_on][goal_state] = real_cost

    #Tell the graph that this node is explored
    my_graph.explored_nodes += [my_graph.transform_to_grid(state_that_has_just_been_trained_on)]

    next_path = my_graph.get_shortest_path(start_state,goal_state) #This outputs a list of tuples

    if len(next_path) <= 2:
        return True


    np2 = my_graph.transform_to_grid(next_path[-2])
    explored_nodes = my_graph.explored_nodes
    embed()
    if np.isin(explored_nodes,np2).all():
        print("I have already explored node ", np2, "so I will delete edge linking it to ",next_path[-3])
        my_graph.delete_edge(next_path[-3],next_path[-2])
        next_point_to_train_on = list(next_path[-3]) #Need to convert tuples to lists
    else:
        print("Node ", np2, "is new! Let's delete its link to the goal and explore it")
        my_graph.delete_edge(next_path[-2], goal_state)
        next_point_to_train_on = list(next_path[-2])

    print("Next path found. It is length ", len(next_path), ":")
    print(next_path)

    # return a new starting point in real coordinates
    return next_point_to_train_on, my_graph