import numpy as np

class dynamic_model:
    def __init__(self, dimension):
        self.dimension = dimension

class linear_model(dynamic_model):
    def __init__(self,A,B,noise_mu=0,noise_sigma=0):
        dynamic_model.__init__(self,A.size[1])
        self.A = A
        self.B = B
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma

    def get_optimal_cost_to_go(self,x0,x1):
        return 0

class linear_model_ballxy(linear_model):
    """The system has the particularity of being round which greatly simplifies collision checking.
    In addition, the two first components of the state space are """
    def __init__(self,radius,A,B,noise_mu=0,noise_sigma=0):
        linear_model.__init__(self,A,B,noise_mu,noise_sigma)
        self.radius = radius

    def in_collision(self,x,EV):


