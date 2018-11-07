import gym
import SimpleBox
from SystemModels import Double_Integrator
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt

my_system = Double_Integrator()

env = TimeLimit(SimpleBox.SimpleBoxEnv(my_system), max_episode_steps=200)
env.reset()

#for _ in range(200):
#    u = env.action_space.sample()
#    print('Action:', u)
#    env.step(u) # take a random action

print('This works!')

quad = SimpleBox.SimpleBoxEnv(my_system)
quad.state = quad.start_state
quad.plot_quad_in_map()
plt.plot(range(5),range(5))
plt.savefig('figure_DoubleIntegrator')
