"""Heat equation solver


"""
import numpy as np
import matplotlib.pyplot as plt

class HeatEquation:
    """
    Class that represents a heat equation.
    
    Attributes:
        D (int) : diffusion constant
        f (callable) : initial condition
        l (callable) : left boundary condition
        r (callable) : right boundary condition
        x_int (list) : interval for the x variable
        t_int (list) : interval for the t variable
    """
    def __init__(self, D: int, f: callable, l: callable, r: callable, x_int: list, t_int: list):
        """Initializes an instance of class HeatEquation with attributes.
        
        Attributes:
            D (int) : diffusion constant
            f (callable) : initial condition
            l (callable) : left boundary condition
            r (callable) : right boundary condition
            x_int (list) : interval for the x variable
            t_int (list) : interval for the t variable
        """
        self.D = D
        self.f = f
        self.l = l
        self.r = r
        self.x_int = x_int
        self.t_int = t_int

    def create_init_cond_array(self, x_points: np.ndarray) -> np.ndarray:
        """Method that creates a numerical value for the initial condition evaluetaed in points x-points.
        Parameters:
            x_points (np.ndarray) : a numpy array with points in the x-direction
        Returns:
            initial_condition (np.ndarray) : a numpy array with values for the initial condition"""
        inital_condition = self.f(x_points)
        return inital_condition

    def create_bound_cond_array(self, g: callable, t_points: np.ndarray) -> np.ndarray:
        """Method that evaluates the boundary condition function in a point t and returns the value.
        Parameters:
            t_points (np.ndarray) : a numpy array with points in the t-direction
        Returns:
            boundary_condition (np.ndarray) : a numpy array with values for the boundary condition"""
        boundary_condition = g(t_points)
        return boundary_condition


class ForwardDiff:
    """Class with methods that solves the heat equation with the forward difference method.
    
    Attributes:
        eq (HeatEquation) : an instance of the class HeatEquation
        m (int) : dimension of the solution matrix
        n (int) : dimension of the time vector
        h (float) : step size in the x-direction
        k (float) : step size in the t-direction
    """
    h = None
    k = None
    
    def __init__(self, equation: HeatEquation, M: int, N: int):
        """Initializes an instance of class ForwardDiff with attributes.
        
        Attributes:
            eq (HeatEquation) : an instance of the class HeatEquation
            m (int) : dimension of the solution matrix
            n (int) : dimension of the time vector
            h (float) : step size in the x-direction
            k (float) : step size in the t-direction
        """
        self.eq = equation
        self.m = M - 1
        self.n = N
        if self.h == None:
            self._update_step_size()

    def _update_step_size(self):
        self.h = (self.eq.x_int[1] - self.eq.x_int[0]) / (self.m + 1)
        self.k = (self.eq.t_int[1] - self.eq.t_int[0]) / (self.n)


    def create_solution_matrix(self) -> np.ndarray:
        """Creates the solution matrix, an (M-1) x (N+1) np.array filled with zeros.
        Returns:
            w (np.array): the solution matrix"""
        w = np.zeros((self.m, self.n + 1))
        return w

    def _initial_condition(self):
        """Methods that creates a numpy array with initial values."""
        x_points = self.eq.x_int[0] + np.arange(1, self.m+1) * self.h
        inital_condition = self.eq.create_init_cond_array(x_points)
        return inital_condition
    
    def add_initial_condition(self, w: np.ndarray) -> np.ndarray:
        """Adds the initial condition as a column vector to the solution matrix.
        Parameters:
            w (np.ndarray) : solution matrix without initial values
        Returns:
            w (np.ndarray) : solution matrix with added initial condition"""
        w[:,0] = self._initial_condition()
        return w

    def _boundary_condition(self, g):
        t_points = self.eq.t_int[0] + np.arange(self.n+1) * self.k
        boundary_condition = self.eq.create_bound_cond_array(g, t_points)
        return boundary_condition
    
    def add_boundary_conditions(self, w: np.ndarray) -> np.ndarray:
        """Expands the solution matrix and adds the left and right boundary conditions as first and last column.
        Parameters:
            w (np.ndarray) : solution matrix with solution
        Returns:
            w_comp (np.ndarray) : expanded solution matrix with added boundary conditions"""
        w_comp = np.zeros((self.m + 2, self.n+1))
        w_comp[0, :] = self._boundary_condition(self.eq.l)
        w_comp[1:-1, :] = w
        w_comp[-1, :] = self._boundary_condition(self.eq.r)

        return w_comp
    
    def _sigma(self):
        sigma = self.eq.D * self.k / (self.h ** 2)
        return sigma

    def _coefficent_matrix(self): 
        sigma = self._sigma()
        main_diagonal = (1 - 2*sigma)*np.ones(self.m)
        super_diagonal = sigma * np.ones(self.m - 1)

        A = np.diag(main_diagonal) + np.diag(super_diagonal, 1) + np.diag(super_diagonal, -1)
        return A
    
    def one_step(self, w: np.ndarray, j: int) -> np.ndarray:
        """Takes one step with the forward difference method. Returns the updated solution matrix.
        Parameters:
            w (np.array): solution matrix to update
            j (int): current column of solution matrix
        Returns:
            updated solution matrix"""

        left_side = self._boundary_condition(self.eq.l)
        right_side = self._boundary_condition(self.eq.r)

        v = np.concatenate(([left_side[j]], np.zeros(self.m - 2), [right_side[j]]))
        b = self._sigma() * v

        A = self._coefficent_matrix()
        return A @ w[:, j] + b

    def solve(self) -> np.ndarray:
        """Solves the equation using forward differences. Initializes the solution matrix, 
        sets the initial conditions and updates the solution. Inserts the boundary values.
        Returns:
            w_comp (n.array): complete solution"""
        w = self.create_solution_matrix()

        w = self.add_initial_condition(w)

        for j in range(self.n):
            w[:, j+1] = self.one_step(w, j)

        w_comp = self.add_boundary_conditions(w)
        
        return w_comp
    

def main():
    D = 1
    f = lambda x: np.sin(2*np.pi*x)**2
    l = lambda t: 0*t
    r = lambda t: 0*t

    x = np.array([0, 1])
    t = np.array([0, 0.5])
    heat_equation = HeatEquation(D, f, l, r, x, t)

    M = 10
    N = 100
    solver = ForwardDiff(heat_equation, M, N)
    solution = solver.solve()

    x = np.linspace(x[0], x[1], solver.m + 2)
    t = np.linspace(t[0], t[1], solver.n + 1)
    T, X = np.meshgrid(t, x)

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, solution, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlim(-0.5, 1)
    ax.view_init(30, 60)
    plt.show()

if __name__ == '__main__':
    main()