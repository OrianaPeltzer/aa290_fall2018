import numpy as np

from utils import *
from curriculum import random, update_backward_reachable_set, sample_from_backward_reachable_set, backward_reachable, graph, graph_fmt,graph_fmt_car
from Graph import Graph
from problem import Problem
from rl import ppo
from collections import defaultdict
from data_logger import DataLogger
from time import strftime
from random_utils import fixed_random_seed
import pickle
import dubins
import matplotlib.pyplot as plt

from IPython import embed

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type", help="what kind of curriculum to employ",
                    type=str)
parser.add_argument("--debug", help="whether to print loads of information and create plots or not",
                    action="store_true")
parser.add_argument("--brs_sample", help="how to sample from a backreachable set",
                    type=str, default='contour_edges')
parser.add_argument("--run_name", help="name of run that determines where outputs are saved",
                    type=str, default=None)
parser.add_argument("--zero_goal_v", help="whether to make the goal have zero initial velocity",
                    action="store_true")
parser.add_argument("--finish_threshold", help="what fraction of starts must be successful to finish training.",
                    type=float, default=0.95)
parser.add_argument("--seed", help="what seed to use for the random number generators.",
                    type=int, default=2018)
parser.add_argument("--disturbance", help="what disturbance to use in the gym environment.",
                    type=str, default=None)
parser.add_argument("--gym_env", help="which gym environment to use.",
                    type=str, default='DrivingOrigin-v0')
parser.add_argument("--hover_at_end", help="whether to null velocity and rates at the end",
                    action="store_true")
parser.add_argument("--variation", help="what variation to use",
                    type=int, default=None)
parser.add_argument("--density_vector", help="How densely to fill the state space using the graph?",
                    type=list, default=[.1,.1,np.pi/12.,.1,.05])

args = parser.parse_args()

if args.type in ['backreach', 'random', 'ppo_only','graph','graph_fmt','graph_fmt_car']:
    if args.run_name is not None:
        run_dir = args.run_name
    else:
        run_dir = args.type
else:
    parser.error('"%s" is not in ["backreach", "random", "ppo_only","graph","graph_fmt","graph_fmt_car"]!' % args.type);

RUN_DIR = os.path.join(os.getcwd(), 'runs', args.gym_env + '_' + run_dir + '_' + strftime('%d-%b-%Y_%H-%M-%S'))
FIGURES_DIR = os.path.join(RUN_DIR, 'figures')
DATA_DIR = os.path.join(RUN_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')

if args.gym_env == 'DrivingOrigin-v0':
    X_IDX = 0
    Y_IDX = 1
    if args.type in ['backreach', 'random', 'ppo_only']:
        from backreach.car5d_interface import Car5DBackreachEngine as BackreachEngine
elif args.gym_env == 'PlanarQuad-v0':
    X_IDX = 0
    Y_IDX = 2
    from backreach.quad6d_interface import Quad6DBackreachEngine as BackreachEngine
elif args.gym_env == "SimpleBox-v0":
    X_IDX = 0
    Y_IDX = 2

def train_step(weighted_start_states, policy, train_algo, problem,
               num_ppo_iters=20):

    problem.start_state_dist = weighted_start_states
    if problem.env_name == 'PlanarQuad-v0':
        gam = 0.99
    else:
        gam = 0.998

    return train_algo.train(problem, policy, min_cost=problem.min_cost, 
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2, entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, 
                            optim_batchsize=64, gamma=gam, lam=0.95,
                            max_iters=num_ppo_iters)


def train(problem,              # Problem object, describing the task.
          initial_policy,       # Initial policy.
          goal_state,           # Goal state (s^g).
          full_start_dist,      # Full start state distribution (rho_0).
          N_new=200, N_old=100, # Num new and old start states to sample.
          R_min=0.5, R_max=1.0, # Reward bounds defining what a "good" start state is.
          num_iters=100,        # Number of iterations of training to run.
          num_ppo_iters=20,     # Number of iterations of PPO training to run per train step.
          curriculum_strategy=random, 
          train_algo=ppo, 
          start_distribution=uniform,
          debug=False,
          density_vector=(0.2,0.1,0.2,0.1)):

    data_logger = DataLogger(col_names=['overall_iter', 'ppo_iter', 
                                        'overall_perf', 'overall_area', 
                                        'ppo_perf', 'ppo_lens', 'ppo_rews'],
                             filepath=os.path.join(DATA_DIR, 'data_%s.csv' % args.type),
                             data_dir=DATA_DIR)

    print(locals(), flush=True);
    if curriculum_strategy in [random]:
        return train_pointwise(**locals());
    elif curriculum_strategy in [backward_reachable]:
        return train_gridwise(**locals());
    elif curriculum_strategy is None:
        return train_ppo_only(**locals());
    elif curriculum_strategy in [graph]:
        return train_graph(fmt=False,**locals())
    elif curriculum_strategy in [graph_fmt]:
        return train_graph(fmt=True,**locals())
    elif curriculum_strategy in [graph_fmt_car]:
        return train_graph_car(fmt=True,**locals())
    else:
        raise ValueError("You passed in an unknown curriculum strategy!");


# "PPO only" because we're not doing any curriculum at all.
def train_ppo_only(**kwargs):
    """Train a policy with no curriculum strategy, only the training algo.
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    pi_i = initial_policy
    overall_perf = list()
    ppo_lens, ppo_rews = list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);
    while i < num_iters*num_ppo_iters:
    # while perf_metric < args.finish_threshold and i < num_iters*num_ppo_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})
        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(full_start_dist, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        overall_perf.append(perf_metric)

        data_logger.add_rows({'overall_perf': [perf_metric], 'ppo_perf': [perf_metric], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                              update_indices=['overall_iter', 'ppo_iter'])

        print('[Overall Iter %d]: perf_metric = %.2f' % (i, perf_metric));

        # Incrementing our algorithm's loop counter.
        # Here, these are the same since the algorithm is PPO itself.
        ppo_iter_count += num_ppo_iters;
        i += num_ppo_iters;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    return pi_i


def train_graph(fmt=False,**kwargs):
    """Train a policy with the graph curriculum
        strategy,
        a specific policy training method (trpo, ppo, etc), and start state sampling
        strategy (uniform, weighted by value function, etc).
        """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];
    density_vector = kwargs["density_vector"]
    k0 = 0 #Default value just to avoid bugs but it is automatically set to 300 in the fmt case
    test_name = "Car_FMT_dubins_arclength_sq_1"

    # ---------------------------- GRAPH CREATION --------------------------- #
    print("Greating Graph")

    environment = problem.environment
    graph_dimension = environment.system.state_dimensions
    start_state = environment.start_state
    goal_state = environment.goal_state

    #Here is where we create the graph ---------------

    my_graph = Graph(graph_dimension)
    limits = environment.state_space_limits_np
    if not fmt:
        print("We will generate a dumb grid.")
        my_graph.fill_graph_gridwise(limits, density_vector, environment)
        with open('my_graph_3.pk1','wb') as output:
            pickle.dump(my_graph, output, pickle.HIGHEST_PROTOCOL)
        print("Saved new graph.")
    else:
        print("FMT will be used to generate a graph.")
        # FMT parameters
        k0 = 20  # Not From paper, > 3**d * exp(1+1/d) took too much time
        my_graph.fill_graph_fmtwise(limits, density_vector, environment,k0)
        with open('my_graph_fmt_car1.pk1', 'wb') as output:
            pickle.dump(my_graph, output, pickle.HIGHEST_PROTOCOL)
        print("Saved new graph.")


    #Here is where we load the graph -----------------
    #if not fmt:
    #    with open('my_graph_3.pk1','rb') as input:
    #        my_graph = pickle.load(input)
    #else:
    #    k0 = 20
    #    with open('my_graph_fmt_newSSM.pk1', 'rb') as input:
    #        my_graph = pickle.load(input)

    print("Graph created. Now finding shortest path:")
    print(start_state)
    print(goal_state)

    #embed()
    my_graph.g[tuple(my_graph.transform_to_grid(goal_state))] = {} #Only keep this while you load, else change graph.py
    first_path,first_path_in_grid = my_graph.get_shortest_path(start_state,goal_state)
    first_state_to_train = list(first_path[-2])
    first_state_to_train_in_grid = list(first_path_in_grid[-2])
    state_that_has_just_been_trained_on_in_grid = first_state_to_train_in_grid
    print("")
    print("Found first path to train on:")
    print(first_path)
    print("")
    my_graph.delete_edge_in_grid(first_state_to_train_in_grid,my_graph.transform_to_grid(goal_state)) #Once you consider an edge for training, delete it

    # ----------------------------------------------------------------------- #

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]

    old_starts = [first_state_to_train]
    starts = [first_state_to_train]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);

    new_starts = np.array(starts) #to avoid confusion since we update the curriculum strategy at the end of the loop

    full_start_dist = list(zip([start_state], uniform([start_state])))

    print("\n Starting to train \n")

    finished = False

    plot_first_dubins_path(first_path,environment.turning_radius,test_name)
    embed()

    while i < 350:
        # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})


        starts = sample(new_starts, size=N_new)

        print("\n The new point to train on is:")
        print(new_starts)

        rho_i = list(zip(starts, start_distribution(starts)))

        #embed()

        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem,
                                                                   num_ppo_iters=num_ppo_iters)

        starts = np.array(starts)
        data_logger.save_to_npy('curr_starts', starts);

        all_starts.extend(starts)
        total_unique_starts = len(dedupe_list_of_np_arrays(starts))
        starts = select(starts, rewards_map, R_min, R_max, problem)
        successful_starts = len(starts)
        pct_successful = float(successful_starts) / total_unique_starts;
        old_starts.extend(starts)

        ppo_perf.append(pct_successful * 100.)
        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        data_logger.save_to_npy('all_starts', np.array(all_starts));
        data_logger.save_to_npy('old_starts', np.array(old_starts));
        data_logger.save_to_npy('selected_starts', starts);
        data_logger.save_to_npy('new_starts', np.array(new_starts));
        #data_logger.save_to_npy('from_replay', from_replay);

        ppo_iter_count += num_ppo_iters;

        #embed()

        perf_metric = evaluate(pi_i,
                               full_start_dist,
                               problem,
                               debug=debug,
                               figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        #if type(perf_metric) != float:
        #    perf_metric = 0.

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))

        overall_perf.append(perf_metric)
        overall_area.append(area_coverage * 100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage],
                              'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                             update_indices=['ppo_iter'])

        print(
            '[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage * 100.));




        state_that_has_just_been_trained_on = new_starts[0]
        print("")
        print("Finished training iteration ", i, ". Now finding new shortest path using updated cost.")
        new_starts, new_starts_in_grid, my_graph, = curriculum_strategy(state_that_has_just_been_trained_on, state_that_has_just_been_trained_on_in_grid, start_state, goal_state, my_graph, pct_successful,ep_mean_rews,environment,k0)
        state_that_has_just_been_trained_on_in_grid = new_starts_in_grid
        my_graph.plot_graph_last(environment,i,test_name)
        print("Saved progression plot.")

        if new_starts == True:
            data_logger.save_to_file();
            pi_i.save_model(MODEL_DIR, iteration=i+1);
            print("The algorithm terminated since we reached the start state!")
            break

        new_starts = np.array(new_starts).reshape((1,my_graph.dimension))

        #embed()

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    with open('trained_graph_start41.pkl', 'wb') as output:
        pickle.dump(my_graph,output,pickle.HIGHEST_PROTOCOL)

    #my_graph.plot_graph_history(environment)

    return pi_i


def train_graph_car(fmt=False, **kwargs):
    """Train a policy with the graph curriculum
        strategy,
        a specific policy training method (trpo, ppo, etc), and start state sampling
        strategy (uniform, weighted by value function, etc).
        """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];
    density_vector = kwargs["density_vector"]
    k0 = 0  # Default value just to avoid bugs but it is automatically set to 300 in the fmt case
    test_name = "Car_FMT_dubins_newcost"

    # ---------------------------- GRAPH CREATION --------------------------- #
    print("Greating Graph")

    environment = problem.environment
    graph_dimension = environment.system.state_dimensions
    start_state = environment.start_state
    goal_state = environment.goal_state

    # Here is where we create the graph ---------------

    my_graph = Graph(graph_dimension)
    limits = environment.state_space_limits_np
    if not fmt:
        print("We will generate a dumb grid.")
        my_graph.fill_graph_gridwise(limits, density_vector, environment)
        with open('my_graph_3.pk1', 'wb') as output:
            pickle.dump(my_graph, output, pickle.HIGHEST_PROTOCOL)
        print("Saved new graph.")
    else:
        print("FMT will be used to generate a graph.")
        # FMT parameters
        k0 = 20  # Not From paper, > 3**d * exp(1+1/d) took too much time
        my_graph.fill_graph_fmtwise(limits, density_vector, environment, k0)
        with open('my_graph_fmt_car1.pk1', 'wb') as output:
            pickle.dump(my_graph, output, pickle.HIGHEST_PROTOCOL)
        print("Saved new graph.")

    # Here is where we load the graph -----------------
    # if not fmt:
    #    with open('my_graph_3.pk1','rb') as input:
    #        my_graph = pickle.load(input)
    # else:
    #    k0 = 20
    #    with open('my_graph_fmt_newSSM.pk1', 'rb') as input:
    #        my_graph = pickle.load(input)

    print("Graph created. Now finding shortest path:")
    print(start_state)
    print(goal_state)

    # embed()
    my_graph.g[
        tuple(my_graph.transform_to_grid(goal_state))] = {}  # Only keep this while you load, else change graph.py
    first_path, first_path_in_grid = my_graph.get_shortest_path(start_state, goal_state)
    first_state_to_train = list(first_path[-2])
    first_state_to_train_in_grid = list(first_path_in_grid[-2])
    state_that_has_just_been_trained_on_in_grid = first_state_to_train_in_grid
    print("")
    print("Found first path to train on:")
    print(first_path)
    print("")
    my_graph.delete_edge_in_grid(first_state_to_train_in_grid, my_graph.transform_to_grid(
        goal_state))  # Once you consider an edge for training, delete it

    # ----------------------------------------------------------------------- #

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]

    old_starts = [first_state_to_train]
    starts = [first_state_to_train]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);

    new_starts = np.array(starts)  # to avoid confusion since we update the curriculum strategy at the end of the loop

    full_start_dist = list(zip([start_state], uniform([start_state])))

    print("\n Starting to train \n")

    finished = False


    step_size = .15

    index_in_path = 0  # For path decomposition
    path_dubins = dubins.shortest_path((first_state_to_train[0], first_state_to_train[1], first_state_to_train[2]),
                                       (goal_state[0], goal_state[1], goal_state[2]), 1.)
    current_path, _ = path_dubins.sample_many(step_size)
    new_starts[0][0] = current_path[-2][0]
    new_starts[0][1] = current_path[-2][1]
    new_starts[0][2] = current_path[-2][2]

    embed()

    while i < 350:
        # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})

        starts = sample(new_starts, size=N_new)

        print("\n The new point to train on is:")
        print(new_starts)

        rho_i = list(zip(starts, start_distribution(starts)))

        # embed()

        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem,
                                                                   num_ppo_iters=num_ppo_iters)

        starts = np.array(starts)
        data_logger.save_to_npy('curr_starts', starts);

        all_starts.extend(starts)
        total_unique_starts = len(dedupe_list_of_np_arrays(starts))
        starts = select(starts, rewards_map, R_min, R_max, problem)
        successful_starts = len(starts)
        pct_successful = float(successful_starts) / total_unique_starts;
        old_starts.extend(starts)

        ppo_perf.append(pct_successful * 100.)
        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        data_logger.save_to_npy('all_starts', np.array(all_starts));
        data_logger.save_to_npy('old_starts', np.array(old_starts));
        data_logger.save_to_npy('selected_starts', starts);
        data_logger.save_to_npy('new_starts', np.array(new_starts));
        # data_logger.save_to_npy('from_replay', from_replay);

        ppo_iter_count += num_ppo_iters;

        # embed()

        perf_metric = evaluate(pi_i,
                               full_start_dist,
                               problem,
                               debug=debug,
                               figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        # if type(perf_metric) != float:
        #    perf_metric = 0.

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))

        overall_perf.append(perf_metric)
        overall_area.append(area_coverage * 100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage],
                              'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                             update_indices=['ppo_iter'])

        print(
            '[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage * 100.));

        # embed()
        index_in_path += 1
        if index_in_path == len(current_path) - 1:
            index_in_path = 0
            state_that_has_just_been_trained_on = new_starts[0]
            print("")
            print("Finished training iteration ", i, ". Now finding new shortest path using updated cost.")
            new_starts, new_starts_in_grid, my_graph, current_path = curriculum_strategy(
                state_that_has_just_been_trained_on, state_that_has_just_been_trained_on_in_grid, start_state,
                goal_state, my_graph, pct_successful, ep_mean_rews, step_size,environment, k0)
            state_that_has_just_been_trained_on_in_grid = new_starts_in_grid
            my_graph.plot_graph_last(environment, i, test_name)

            if new_starts != True:
                new_starts[0][0] = current_path[-2 - index_in_path][0]
                new_starts[0][1] = current_path[-2 - index_in_path][1]
                new_starts[0][2] = current_path[-2 - index_in_path][2]
                print("Saved progression plot.")
        else:
            new_starts[0][0] = current_path[-2-index_in_path][0]
            new_starts[0][1] = current_path[-2-index_in_path][1]
            new_starts[0][2] = current_path[-2-index_in_path][2]

        if type(new_starts) == type(True):
            data_logger.save_to_file();
            pi_i.save_model(MODEL_DIR, iteration=i + 1);
            print("The algorithm terminated since we reached the start state!")
            break

        new_starts = np.array(new_starts).reshape((1, my_graph.dimension))

        # embed()

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    with open('trained_graph_start41.pkl', 'wb') as output:
        pickle.dump(my_graph, output, pickle.HIGHEST_PROTOCOL)

    # my_graph.plot_graph_history(environment)

    return pi_i


# "Pointwise" because the unit of reachibility here is explicit states,
# i.e. "points in the state space." Our random baseline is point-based.
def train_pointwise(**kwargs):
    """Train a policy with a specific curriculum 
    strategy (random, etc), 
    policy training method (trpo, ppo, etc), and start state sampling 
    strategy (uniform, weighted by value function, etc).
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]
    
    old_starts = [goal_state]
    starts = [goal_state]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
    perf_metric = 0.0
    i = 0
    ppo_iter_count = 0;
    pi_i.save_model(MODEL_DIR, iteration=i);
    while i < num_iters:
    # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True)
        data_logger.update_indices({"overall_iter": i, "ppo_iter": ppo_iter_count})

        new_starts = curriculum_strategy(starts, N_new, problem)

        from_replay = sample(old_starts, size=N_old)
        starts = new_starts + from_replay

        rho_i = list(zip(starts, start_distribution(starts)))
        pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

        data_logger.save_to_npy('curr_starts', starts);

        all_starts.extend(starts)
        total_unique_starts = len(dedupe_list_of_np_arrays(starts))
        starts = select(starts, rewards_map, R_min, R_max, problem)
        successful_starts = len(starts)
        pct_successful = float(successful_starts)/total_unique_starts;
        old_starts.extend(starts)

        ppo_perf.append(pct_successful*100.)
        ppo_lens.extend(ep_mean_lens)
        ppo_rews.extend(ep_mean_rews)

        data_logger.save_to_npy('all_starts', all_starts);
        data_logger.save_to_npy('old_starts', old_starts);
        data_logger.save_to_npy('selected_starts', starts);
        data_logger.save_to_npy('new_starts', new_starts);
        data_logger.save_to_npy('from_replay', from_replay);

        ppo_iter_count += num_ppo_iters;

        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))

        overall_perf.append(perf_metric)
        overall_area.append(area_coverage*100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage], 
                              'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews},
                              update_indices=['ppo_iter'])

        print('[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage*100.));

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    return pi_i


# "Gridwise" because the unit of reachibility here is grids of the state space.
# Our backward reachibility method is grid-based.
def train_gridwise(**kwargs):
    """Train a policy with a specific curriculum 
    strategy (backward reachable, etc), 
    policy training method (trpo, ppo, etc), and start state sampling 
    strategy (uniform, weighted by value function, etc).
    """
    problem = kwargs["problem"];
    initial_policy = kwargs["initial_policy"];
    goal_state = kwargs["goal_state"];
    full_start_dist = kwargs["full_start_dist"];
    N_new, N_old = kwargs["N_new"], kwargs["N_old"];
    R_min, R_max = kwargs["R_min"], kwargs["R_max"];
    num_iters = kwargs["num_iters"];
    num_ppo_iters = kwargs["num_ppo_iters"];
    curriculum_strategy = kwargs["curriculum_strategy"];
    train_algo = kwargs["train_algo"];
    start_distribution = kwargs["start_distribution"];
    debug = kwargs["debug"];
    data_logger = kwargs["data_logger"];

    # Keyword arguments for the curriculum strategy.
    curric_kwargs = defaultdict(lambda: None)
    curric_kwargs["debug"] = debug
    if curriculum_strategy == backward_reachable:
        br_engine = BackreachEngine()
        br_engine.reset_variables(problem, os.path.join(FIGURES_DIR, ''))
        curric_kwargs['br_engine'] = br_engine
        curric_kwargs['curr_train_iter'] = 0
        curric_kwargs['brs_sample'] = args.brs_sample
        curric_kwargs['variation'] = args.variation
        curric_kwargs['problem'] = problem

    # Not used in algorithm, only for visualization.
    all_starts = [goal_state]
    
    old_starts = [goal_state]
    starts = [goal_state]
    pi_i = initial_policy
    overall_perf, overall_area = list(), list()
    perf_metric = 0.0
    i = 0
    pi_i.save_model(MODEL_DIR, iteration=i);



    while i < num_iters:
    # while perf_metric < args.finish_threshold and i < num_iters:
        print('Training Iteration %d' % i, flush=True);
        data_logger.update_indices({"overall_iter": i})

        if 'curr_train_iter' in curric_kwargs:
            curric_kwargs['curr_train_iter'] = i;

        # I've split apart the following call into two separate ones.
        # new_starts = curriculum_strategy(starts, N_new, problem, **curric_kwargs)
        if curriculum_strategy == backward_reachable:
            update_backward_reachable_set(starts, **curric_kwargs);
            data_logger.save_to_npy('brs_targets', starts);

        pct_successful = 0.0;
        iter_count = 0;
        ppo_perf, ppo_lens, ppo_rews = list(), list(), list()
        # Think of this as "while (haven't passed this grade)"
        while pct_successful < 0.5:
            data_logger.update_indices({"ppo_iter": iter_count})

            if curriculum_strategy == backward_reachable:
                new_starts = sample_from_backward_reachable_set(N_new, **curric_kwargs);

            if args.variation == 2 and br_engine.check_membership(np.array([problem.env.unwrapped.start_state])):
                print('Variation 2 condition train.py!', flush=True)
                from_replay = list()
                starts = [problem.env.unwrapped.start_state]
            else:
                from_replay = sample(old_starts, size=N_old)
                starts = new_starts + from_replay

            rho_i = list(zip(starts, start_distribution(starts)))
            pi_i, rewards_map, ep_mean_lens, ep_mean_rews = train_step(rho_i, pi_i, train_algo, problem, num_ppo_iters=num_ppo_iters)

            data_logger.save_to_npy('curr_starts', starts);

            all_starts.extend(starts)
            total_unique_starts = len(dedupe_list_of_np_arrays(starts))
            starts = select(starts, rewards_map, R_min, R_max, problem)
            successful_starts = len(starts)
            pct_successful = float(successful_starts)/total_unique_starts;

            ppo_perf.append(pct_successful*100.)
            ppo_lens.extend(ep_mean_lens)
            ppo_rews.extend(ep_mean_rews)

            data_logger.add_rows({'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews}, update_indices=['ppo_iter'])

            data_logger.save_to_npy('all_starts', all_starts);
            data_logger.save_to_npy('old_starts', old_starts);
            data_logger.save_to_npy('selected_starts', starts);
            data_logger.save_to_npy('new_starts', new_starts);
            data_logger.save_to_npy('from_replay', from_replay);

            iter_count += num_ppo_iters;
            print('[PPO Iter %d]: %.2f%% Successful Starts (%d / %d)' % (iter_count, pct_successful*100., successful_starts, total_unique_starts));

        # This final update is so we get the last iter_count correctly after jumping out of the while loop.
        data_logger.update_indices({"ppo_iter": iter_count})

        # Ok, we've graduated!
        old_starts.extend(starts)
        perf_metric = evaluate(pi_i, 
                             full_start_dist, 
                             problem, 
                             debug=debug, 
                             figfile=os.path.join(FIGURES_DIR, 'eval_iter_%d' % i))

        # Format is (min_x, max_x, min_y, max_y)
        all_starts_bbox = bounding_box(all_starts)
        min_x = problem.state_space.low[X_IDX]
        max_x = problem.state_space.high[X_IDX]
        min_y = problem.state_space.low[Y_IDX]
        max_y = problem.state_space.high[Y_IDX]
        area_coverage = bounding_box_area(all_starts_bbox) / bounding_box_area((min_x, max_x, min_y, max_y))
        
        overall_perf.append(perf_metric)
        overall_area.append(area_coverage*100.)

        data_logger.add_rows({'overall_perf': [perf_metric], 'overall_area': [area_coverage]})

        print('[Overall Iter %d]: perf_metric = %.2f | Area Coverage = %.2f%%' % (i, perf_metric, area_coverage*100.));

        # Incrementing our algorithm's loop counter.
        i += 1;

        data_logger.save_to_file();
        pi_i.save_model(MODEL_DIR, iteration=i);

    # Done!
    if curriculum_strategy == backward_reachable:
        br_engine.stop();
        del br_engine;

    return pi_i


def plot_first_dubins_path(first_path,TR,test_name):
    Xs = [elt[0] for elt in first_path]
    Ys = [elt[1] for elt in first_path]
    ths = [elt[2] for elt in first_path]

    ax = plt.gca()
    r = plt.Rectangle((-0.2, -0.2), 0.4,
                      0.4,
                      color='g', alpha=0.3, hatch='/')
    ax.add_artist(r)

    for k in range(len(first_path)-1):
        dp = dubins.shortest_path((Xs[k],Ys[k],ths[k]),(Xs[k+1],Ys[k+1],ths[k+1]),TR)
        cf,_ = dp.sample_many(0.01)
        xs = [elt[0] for elt in cf]
        ys = [elt[1] for elt in cf]
        plt.plot(xs,ys,color='g')
    plt.plot(Xs, Ys, color='k')

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.savefig('plots/dubins_first_path.png')
    return

if __name__ == '__main__':
    # Using context managers for random seed management.
    with fixed_random_seed(args.seed):
        maybe_mkdir(RUN_DIR);
        maybe_mkdir(DATA_DIR);
        maybe_mkdir(FIGURES_DIR);
        maybe_mkdir(MODEL_DIR);

        # Keeping dist same across runs for comparison.
        with fixed_random_seed(2018):
            problem = Problem(args.gym_env, 
                              zero_goal_v=args.zero_goal_v, 
                              disturbance=args.disturbance)

            if args.gym_env == 'DrivingOrigin-v0':
                num_iters = 100
                print('Goal State:', problem.goal_state, flush=True)
                full_starts = problem.sample_behind_goal(problem.goal_state,
                                                         num_states=100, 
                                                         zero_idxs=[3, 4])
            elif args.gym_env == 'PlanarQuad-v0':
                num_iters = 40
                full_starts = [problem.env.unwrapped.start_state]
                problem.env.unwrapped.set_hovering_goal(args.hover_at_end)

            elif args.gym_env == 'SimpleBox-v0':
                num_iters = 10
                print('Goal State:', problem.goal_state,flush=True)
                full_starts = [problem.env.unwrapped.start_state]

        full_start_dist = list(zip(full_starts, uniform(full_starts)))

        ppo.create_session(num_cpu=1)
        initial_policy = ppo.create_policy('pi', problem)
        ppo.initialize()

        if args.type == 'random':
            curr_strategy = random
        elif args.type == 'backreach':
            curr_strategy = backward_reachable
        elif args.type == 'ppo_only':
            curr_strategy = None
        elif args.type == 'graph':
            curr_strategy = graph
        elif args.type == 'graph_fmt':
            curr_strategy = graph_fmt
        elif args.type == 'graph_fmt_car':
            curr_strategy = graph_fmt_car
        else:
            raise ValueError("%s is an unknown curriculum strategy!" % args.type);

        trained_policy = train(problem=problem,
                               num_iters=num_iters,
                               R_min=problem.R_min, R_max=problem.R_max,
                               initial_policy=initial_policy,
                               goal_state=problem.goal_state,
                               full_start_dist=full_start_dist,
                               curriculum_strategy=curr_strategy,
                               debug=args.debug,
                               density_vector=args.density_vector)

        trained_policy.save_model(MODEL_DIR);
        print('Done training!');
