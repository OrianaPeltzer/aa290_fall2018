"""
Nonlinear planar quad model with laser sensors implemented by 
James Harrison and Apoorva Sharma

Implements a 6D state space + 14D observation space where the agent drives to the origin.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#from IPython import embed

from SystemModels import Double_Integrator

logger = logging.getLogger(__name__)

class SimpleBoxEnv(gym.Env):
    """This implements a simple box environment with a 2D stable controllable MIMO system.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,system):
        self.m = 1.25
        self.Cd_v = 0.25
        self.Cd_phi = 0.02255
        self.Iyy = 0.03
        self.g = 9.81
        self.l = 0.2
        self.Tmax = 1.00*self.m*self.g
        self.Tmin = 0

        self.num_obst = 3
        self.num_sensors = 8

        self.control_cost = 0.01
        self.goal_bonus = 1000
        self.collision_cost = -2*200*self.control_cost*self.Tmax**2


        self.system = system

        # What qualifies as a "success" such that we select it when expanding states?
        # This is a normalized value, akin to dividing the reward by the absolute of the 
        # min_cost and then shifting it so that all values are positive between 0 and 1.
        # This does NOT affect PPO, only our selection algorithm after.
        self.R_min = 0.5
        self.R_max = 1.0

        self.quad_rad = self.l

        #bounding box
        self.x_upper = 5.
        self.x_lower = 0.
        self.y_upper = 5.
        self.y_lower = 0.

        #other state bounds
        self.v_limit = 2.
        self.phi_limit = 5.
        self.omega_limit = np.pi/6.

        #goal region
        # Have no fear, goal_state isn't used anywhere, 
        # it's just for compatibility.
        # x, vx, y, vy, phi, omega
        self.goal_state = self.system.create_xy_goal(np.array([4.,4.]))

        self.xg_lower = 4.
        self.yg_lower = 4.
        self.xg_upper = 5.
        self.yg_upper = 5.
        self.g_vel_limit = 0.25
        self.g_phi_limit = np.pi/6.
        # This isn't actually used for goal pos calculations, 
        # but for backreachability
        self.g_pos_radius = 0.1

        # After defining the goal, create the obstacles.
        self._generate_obstacles()

        self.dt = 0.1

        self.start_state = self.system.create_start_state_xxdyyd(np.array([4.,0.,1.,0.]))

        self.min_cost = self.collision_cost - 2*200*self.control_cost*self.Tmax**2 

        high_state, low_state = self.system.create_limit_states_xylim(self.x_upper,self.x_lower,self.y_upper,self.y_lower)

        high_obsv, low_obsv = self.system.create_observation_limits(self.x_upper,self.x_lower,self.y_upper,self.y_lower)

        high_actions, low_actions = self.system.create_action_limits()


        self.action_space = spaces.Box(low=low_actions,high=high_actions)
        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsv, high=high_obsv)
        self.state_space_limits_np = np.array([low_state,high_state]).T

        self.seed(2015)
        self.viewer = None


    def map_action(self, action):
        return [ self.Tmin + (0.5 + a/6.0)*(self.Tmax - self.Tmin) for a in action ]

    def set_disturbance(self, disturbance_str):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def x_dot(self,z,u):
        return self.system.x_dot(z,u)


    def _generate_obstacles(self):
        #currently, making the obstacle placement deterministic so that we guarantee feasibility
        # Temporarily removing obstacles.
        self.obst_R = np.array([0.5,1.0,0.5])
        self.obst_X = np.array([4.0,1.0,1.0])
        self.obst_Y = np.array([2.5,1.0,4.0])

    def _in_goal(self, state):
        xq = state[0]
        yq = state[2]

        if (xq < self.xg_upper) and (xq > self.xg_lower) and (yq < self.yg_upper) and (yq > self.yg_lower):
            return True
        else:
            return False


    def plot_quad_in_map(self):
        x = self.state[0]
        y = self.state[2]

        ax = plt.gca()
        for xo,yo,ro in zip(self.obst_X, self.obst_Y, self.obst_R):
            c = plt.Circle((xo,yo),ro, color='black', alpha=1.0)
            ax.add_artist(c)
       
        r = plt.Rectangle((self.xg_lower, self.yg_lower), self.xg_upper-self.xg_lower, self.yg_upper - self.yg_lower, color='g', alpha=0.3, hatch='/')
        ax.add_artist(r)

        plt.plot([x], [y], marker='o', linewidth=2, color='b', markersize=5)

        plt.xlim([self.x_lower, self.x_upper])
        plt.ylim([self.y_lower, self.y_upper])

    def _in_obst(self, state):
        xq = state[0]
        yq = state[2]

        if (xq + self.quad_rad > self.x_upper) or (xq - self.quad_rad < self.x_lower) or (yq + self.quad_rad > self.y_upper) or (yq - self.quad_rad < self.y_lower):
            return True

        for i in range(self.num_obst):
            d = (xq - self.obst_X[i])**2 + (yq - self.obst_Y[i])**2
            r = (self.quad_rad + self.obst_R[i])**2
            if d < r:
                return True

        return False

    def _get_obs(self, state):
        return self.system.get_observation(state)

    def step(self, action):
        #map action is only needed in the quadrotor case since the zero thrust makes everything crash.
        #action = self.map_action(action)

        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action));

        #print("I am in step. Debug me please.")
        #embed()

        #clip actions
        action = np.clip(action,self.Tmin,self.Tmax)

        old_state = np.array(self.state)

        t = np.arange(0, self.dt, self.dt*0.01)

        integrand = lambda x,t: self.x_dot(x, action)

        x_tp1 = odeint(integrand, old_state, t)
        self.state = x_tp1[-1,:] 

        # Be close to the goal and have the desired final velocity.
        reward = - self.control_cost*(action[0]**2 + action[1]**2)
        done = False

        if self._in_goal(self.state):
            reward += self.goal_bonus
            done = True
        # not currently checking along the trajectory for collision violation
        if self._in_obst(self.state):
            reward += self.collision_cost
            done = True

        return self._get_obs(self.state), reward, done, {}

    def reset(self):
        self._generate_obstacles()
        # self._generate_goal() 

        #currently generating static start state
        self.state = self.start_state.copy()
        
        return self._get_obs(self.state)

    def render(self, mode='human', close=False):
        pass
