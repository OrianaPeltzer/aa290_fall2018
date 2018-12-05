import numpy as np


class System():
#General system from which we can create many sub_systems inherited from this class. Here are the necessary functions and attributes
# that are likely to be called upon when using any object of the System class
    def __init__(self,state_dimensions,control_input_dimensions, state_bounds = None, control_bounds = None):
        self.state = np.zeros(state_dimensions)
        self.state_dimensions = state_dimensions
        self.control_input = np.zeros(control_input_dimensions)
        self.control_input_dimensions = control_input_dimensions

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
        return np.dot(self.A,x) + np.dot(self.B,u)

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
        #A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [-3, 1, 2, 3], [2, 1, 0, 0]])
        B = np.array([[0,0],[1,0],[0,0],[0,1]])
        #B = np.array([[0, 0], [0, 0], [1, 2], [0, 2]])
        C = np.eye(4)
        SimpleMIMO.__init__(self,A,B,C)

        #Velocity limits
        self.xd_upper_lim = 1.
        self.yd_upper_lim = 1.
        self.xd_lower_lim = -1.
        self.yd_lower_lim = -1.

        #Control input limits
        self.ux_upper_lim = 5.
        self.uy_upper_lim = 5.
        self.ux_lower_lim = -5.
        self.uy_lower_lim = -5.

        #Q and R for solving optimal control problem
        self.Q = np.eye(self.state_dimensions)
        self.R = np.eye(self.control_input_dimensions)
        self.H = np.eye(self.state_dimensions)

        #Less computations in the solve optimal control function
        self.Rm1 = np.linalg.inv(self.R)
        self.M = np.dot(self.B, self.Rm1).dot(self.B.T)

        #Initialize these (only useful to avoid bugs)
        self.K =False
        self.K_j = False
        #Creates self.K the controller
        self.solve_LQR_K()



    def create_xy_goal(self,goal):
        #The goal is set at goal=[x,y], we choose to set the velocity at exactly 1 to avoid issues.
        return np.array([goal[0],1e-4,goal[1],1e-4])

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

    def solve_LQR_K(self,time_horizon = 0.5):
        K = self.H
        delta_T = 0.001
        print("Finding optimal LQR controller")
        for i in np.mgrid[delta_T:time_horizon:delta_T]:
            K_dot = -self.Q + K.dot(self.M).dot(K.T) - np.dot(K,self.A) - self.A.T.dot(K)
            K = K - delta_T*K_dot

            if np.linalg.norm(K_dot) < 0.05:
                break
        self.K_j = K #This is P (psd), found as solution to finite horizon Riccati equation, verifying that the optimal cost is xT*P*x
        print("Solution to Riccati equation: P=")
        print(self.K_j)
        if np.all(np.linalg.eigvals(self.K_j) >= 0):
            print("P is Positive Semi Definite - OK")
        else:
            print("P is not positive semi definite")
        self.K = -self.Rm1.dot(self.B.T).dot(K)
        print("Found optimal controller. K=")
        print(self.K)
        return

    def solve_optimal_control_cost(self, node_start, node_end):
        return 0.5*(node_start-node_end).T.dot(self.K_j).dot(node_start-node_end)

    def is_feasible_LQR_path(self,node_start,node_end,environment,time_horizon = 0.5):
        delta_T = 0.001
        x = node_start
        for i in np.mgrid[delta_T:time_horizon:delta_T]:
            x_dot = self.x_dot(x, self.K.dot(x-node_end))
            x += delta_T*x_dot
            if (i/delta_T)%10 == 0: #We won't always to the test, only sometimes
                if environment._in_obst(x):
                    return False, np.inf #If the path is infeasible then it has an infinite expected cost
        return True, np.linalg.norm(x-node_end)

