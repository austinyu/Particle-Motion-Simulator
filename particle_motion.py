import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class Particle_Motion:
    """
    This is a class specialized in plotting and animating the motion of particles under 
    the influce of some field. 

    Member Variables
    ----------
    self.experiment : list
        a list of names for each experiment executed. 
    self.positions_dict : dict()
        a dictionary that maps experiment name to its potision vectors 
        These two variables combined act as a ordered dict that sort by 
        experiment execution sequence. 

    Member Methods
    -------
    __init__(self, deq = __EB_field_deq)
    run_experiment(self,
                   name = 'NA', 
                   params = {'charge' : 1,
                            'mass' : 1,
                            'b_field' : lambda x, y, z: [1,0,0],
                            'e_field' : lambda x, y, z: [0,0,0]},
                   initial_state = np.array([0, 0, 0, 1, 1, 0]), # position + velocity states
                   t_final = 30,
                   dt = 0.01)
    plot_positions(self)
    animate_positions(self, fps = 30)
    
    Example 
    ------
demo = Particle_Motion()
demo_params = [("exp1", 
                    {'charge' : 1, 
                        'mass' : 5, 
                        'b_field' : lambda x, y, z: [1,0,0],  
                        'e_field' : lambda x, y, z: [0,0,0]},
                     np.array([0, 0, 0, 1, 1, 0])), 
                ("exp2", 
                    {'charge' : 2,
                        'mass' : 3,
                        'b_field' : lambda x, y, z: [1,0,0],
                        'e_field' : lambda x, y, z: [0,0,0]},
                    np.array([0, 0, 0, 1, 1, 0])),
                ("exp3", 
                    {'charge' : -1,
                        'mass' : 1,
                        'b_field' : lambda x, y, z: [1,0,0],
                        'e_field' : lambda x, y, z: [0,0,0]},
                    np.array([0, 0, 0, 1, 1, 0]))]
for name, params, initial_state in demo_params:
    demo.run_experiment(name, params = params, initial_state = initial_state)
    """
    def __EB_field_deq(time_step, state_vec, charge, mass, b_field, e_field):
        """
        Private default differential equation for particle motion in ExB field
            not callable out side of the class. 

        Parameters
        ----------
        time_step : float64
            the size of time for each step of numerical integration. 
        state_vec : np.ndarray
            a numpy array containing state information for a particle 
            at a particular time. [x, y, z, vx, vy, vz]
        charge : int
            value of charge carried by the particle.
        mass : int 
            mass of the particle
        b_field : Callable function
            a function that maps a set of coordinate to a vector representing 
            the magnetic field vector at that point. 
        e_field : Callable function
            a function that maps a set of coordinate to a vector representing 
            the electrical field vector at that point. 

        Returns
        -------
        np.ndarray
            a numpy array containing a new state information for a particle 
            at the next time step. [x, y, z, vx, vy, vz]

        Raises
        ------
        None
        """
        x, y, z, vx, vy, vz = state_vec
        bx, by, bz = b_field(x, y, z)
        ex, ey, ez = e_field(x, y, z)
        coefficient = charge/mass
        return np.array([vx, vy, vz, coefficient * (vy*bz-vz*by+ex), 
                                     coefficient * (vz*bx-vx*bz+ey), 
                                     coefficient * (vx*bz-vy*bx+ez)])
    def __init__(self, deq = __EB_field_deq):
        """
        Constructor 

        Parameters
        ----------
        deq : Callable function
            the differenntial equation used for solving, plotting and animating 
            the motion of particles. 

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.deq = deq
        self.experiment = []
        self.positions_dict = dict()

    def __solve_deq(self, params, initial_state, t_final, dt):
        """
        Private function that solves deq based on the initial condition given. 

        Parameters
        ----------
        params : dict()
            A dictionary stroing parameters in the numerical integration. 
            params.keys() = ['charge', 'mass', 'b_field',  'e_field']
        initial_state : list
            A list storing initial state vector for a particle. 
            [x, y, z, vx, vy, vz]
        t_final : float64
            Terminating time for integration. 
        dt : float64
            The duration of time for each step of numerical integration

        Returns
        -------
        np.ndarray
            A 2d array that has dimentions (t_final//dt, 3).
            It stores position vector at each time step of integration. 

        Raises
        ------
        None
        """
        integrator = ode(self.deq).set_integrator('dopri5')
        charge, mass = params['charge'], params['mass']
        b_field, e_field = params['b_field'], params['e_field']
        integrator.set_initial_value(initial_state, 0).set_f_params(charge, mass, b_field, e_field)
        positions = []
        while integrator.successful() and integrator.t < t_final:
            integrator.integrate(integrator.t+dt)
            positions.append(integrator.y[:3])
        return np.array(positions)
    def run_experiment(self,
                       name = 'NA', 
                       params = {'charge' : 1,
                                'mass' : 1,
                                'b_field' : lambda x, y, z: [1,0,0],
                                'e_field' : lambda x, y, z: [0,0,0]},
                       initial_state = np.array([0, 0, 0, 1, 1, 0]), # position + velocity states
                       t_final = 30,
                       dt = 0.01):
        """
        Run experiment on the motion of a particle based on the parameters, and initial states 
        give. Resulting position vector will be stored in a member variable dictionary that can 
        be used for plots and animations

        Parameters
        ----------
        name : str = 'NA'
            The name of the experiment. 
        params : dict() = {'charge' : 1,
                            'mass' : 1,
                            'b_field' : lambda x, y, z: [1,0,0],
                            'e_field' : lambda x, y, z: [0,0,0]}
            A dictionary that stores required parameters for integration. 
        initial_state : np.ndarray = np.array([0, 0, 0, 1, 1, 0])
            A list storing initial state vector for a particle. 
            [x, y, z, vx, vy, vz]
        t_final : float64 = 30
            Terminating time for integration. 
        dt : float64  = 0.01
            The duration of time for each step of numerical integration

        Returns
        -------
        None

        Raises
        ------
        KeyError
            when the function is given a previously defined experiment name. 
        """
        if name in self.positions_dict.keys():
            raise KeyError('Experiment name is already taken. Please choose another one.')
        if name == 'NA':
            name = 'Experiment' + str(len(self.positions_dict))
        self.experiment.append(name)
        self.positions_dict[name] = self.__solve_deq(params = params,
                                                  initial_state = initial_state, 
                                                  t_final = t_final,
                                                  dt = dt)
    def plot_positions(self):
        """
        Plot the motion of particles based on member variable self.positions_dict

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for name, positions in self.positions_dict.items():
            ax.plot3D(positions[:,0], 
                      positions[:,1], positions[:,2],
                      label = name)
        ax.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
    def animate_positions(self, fps = 30):
        """
        Animate the motion of particles based on member variable self.positions_dict

        Parameters
        ----------
        fps : int = 30
            frame per second for the animation. 

        Returns
        -------
        plt animation object

        Raises
        ------
        None
        """
        dim = list(self.positions_dict.values())[0].shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        def init():
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            
        def update(frame):
            index = int(dim / fps * frame)
            ax.cla()
            for name, positions in self.positions_dict.items():
                ax.plot3D(positions[:index, 0], 
                          positions[:index, 1], positions[:index, 2],
                          label = name)
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        return FuncAnimation(fig, update, frames=fps,
                            init_func=init, interval=100)