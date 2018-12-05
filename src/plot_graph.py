from Graph import Graph
import SimpleBox
from SystemModels import Double_Integrator
import pickle
from IPython import embed

my_system = Double_Integrator()
environment = SimpleBox.SimpleBoxEnv(my_system)

with open('trained_graph.pk1','rb') as input:
    my_graph = pickle.load(input)

#print(my_graph.explored_nodes_history)
#embed()

my_graph.plot_graph_history(environment)