import numpy as np

class System():
#General system from which we can create many sub_systems inherited from this class. Here are the necessary functions and attributes
# that are likely to be called upon when using any object of the System class
    def __init__(self,state_dimensions,control_input_dimensions, state_bounds = None, control_bounds = None):
        self.state = np.zeros(state_dimensions)
        self.control_input = np.zeros(control_input_dimensions)

    def x_dot(self,x,u):
        return

    def simulate_step(self,xi,u,delta_T=0.01):
        return

class SimpleMIMO(System):
#Simple MIMO system with form x_dot = Ax + Bu, y = Cx, no disturbances.
    def __init__(self,A,B,C):
        state_dimensions = np.shape(A)[0]
        control_input_dimensions = np.shape(B)[1]
        System.__init__(self,state_dimensions,control_input_dimensions)
        self.A = A
        self.B = B
        self.C = C

    def x_dot(self,x,u):
        return self.A*x + self.B*u

    def simulate_step(self,xi,u,delta_T=0.01):
        # Uses Euler's method to step forwards in time and simulate the system
        x_dot = self.x_dot(xi,u)
        xtp1 = x + delta_T*x_dot
        return xtp1

    def get_observation(self,state):
        #y = Cx
        return np.dot(self.C,state)


# -------------------------------------- SPECIFIC EXAMPLES -------------------------------- #

class Double_Integrator(SimpleMIMO):
    #Specific 2D Double integrator with two control inputs
    def __init__(self):
        #This represents a state [x, xd, y, yd]' with state space equations:
        # xd = xd
        # xdd = ux
        # yd = yd
        # ydd = uy
        # y = Ix
        A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
        B = np.array([[0,0],[1,0],[0,0],[0,1]])
        C = np.eye(4)
        SimpleMIMO.__init__(self,A,B,C)

        #Velocity limits
        self.xd_upper_lim = 300.
        self.yd_upper_lim = 300.
        self.xd_lower_lim = -300.
        self.yd_lower_lim = -300.

        #Control input limits
        self.ux_upper_lim = 500.
        self.uy_upper_lim = 500.
        self.ux_lower_lim = -500.
        self.uy_lower_lim = -500.


    def create_xy_goal(self,goal):
        #The goal is set at goal=[x,y], we choose to set the velocity at exactly 1 to avoid issues.
        return np.array([goal[0],1.,goal[1],1.])

    def create_start_state_xxdyyd(self,start):
        #No modifications to make here, start is already in the right form.
        return start

    def create_limit_states_xylim(self,xu,xl,yu,yl):
        #The upper and lower x and y limits come from the environment box.
        #We create velocity limits that depend on our system.
        #Returns high_state,low_state
        return np.array([xu,self.xd_upper_lim,yu,self.yd_upper_lim]),np.array([xl,self.xd_lower_lim,yl,self.yd_lower_lim])

    def create_observation_limits(self,xu,xl,yu,yl):
        #Since y = Ix we just return state limits
        return self.create_limit_states_xylim(xu,xl,yu,yl)

    def create_action_limits(self):
        #Returns high_action, low_action
        return np.array([self.ux_upper_lim,self.uy_upper_lim]), np.array([self.ux_lower_lim,self.uy_lower_lim])

