# Matilda Stoetzer, grudat25, individuellt projekt
import numpy as np

class HeatEquation:
    """
    Class that represents a heat equation.
    
    Attributes:
        D (int) : diffusion constant
        f (callable) : initial condition
        l (callable) : left boundary condition
        r (callable) : right boundary condition
        x_int (np.ndarray) : interval for the x variable
        t_int (np.ndarray) : interval for the t variable
    """
    def __init__(self, D: int, f: callable, l: callable, r: callable, x_int: np.ndarray, t_int: np.ndarray):
        """Initializes an instance of class HeatEquation with attributes.
        
        Attributes:
            D (int) : diffusion constant
            f (callable) : initial condition
            l (callable) : left boundary condition
            r (callable) : right boundary condition
            x_int (np.ndarray) : interval for the x variable
            t_int (np.ndarray) : interval for the t variable
        """
        self._validate_attribute_types(D, f, l, r, x_int, t_int)
        self._validate_attribute_values(D, x_int, t_int)
        
        self.D = D
        self.f = f
        self.l = l
        self.r = r
        self.x_int = x_int
        self.t_int = t_int

# --- public methods ---

    def create_init_cond_array(self, x_points: np.ndarray) -> np.ndarray:
        """Method that creates a numerical value for the initial condition evaluetaed in points x-points.
        Parameters:
            x_points (np.ndarray) : a numpy array with points in the x-direction
        Returns:
            initial_condition (np.ndarray) : a numpy array with values for the initial condition"""
        
        if not isinstance(x_points, (np.ndarray)):
            raise TypeError('The x interval must be a numpy array.')
        
        inital_condition = self.f(x_points)
        return inital_condition


    def create_bound_cond_array(self, g: callable, t_points: np.ndarray) -> np.ndarray:
        """Method that evaluates the boundary condition function in a point t and returns the value.
        Parameters:
            t_points (np.ndarray) : a numpy array with points in the t-direction
        Returns:
            boundary_condition (np.ndarray) : a numpy array with values for the boundary condition"""
        
        if not callable(g):
            raise TypeError('The boundary condition must be of type callable.')
        if not isinstance(t_points, (np.ndarray)):
            raise TypeError('t_points must be a numpy array.')
        
        boundary_condition = g(t_points)
        return boundary_condition
    
# --- private methods ---

    def _validate_attribute_types(self, D: int, f: callable, l: callable, r: callable, x_int: np.ndarray, t_int: np.ndarray):
        """Checks the data types of the parameters passed to the class. Raises TypeError if wrong data type."""
        if not isinstance(D, int):
            raise TypeError('D must be an integer.')
        if not callable(f):
            raise TypeError('The initial condition must be of type callable.')
        if not callable(l):
            raise TypeError('The boundary condition must be of type callable.')
        if not callable(r):
            raise TypeError('The boundary condition must be of type callable.')
        if not isinstance(x_int, np.ndarray):
            raise TypeError('The x interval must be a numpy array.')
        if not isinstance(t_int, np.ndarray):
            raise TypeError('The t interval must be a numpy array.')
     
    def _validate_attribute_values(self, D: int, x_int: np.ndarray, t_int: np.ndarray):
        """Check if interval for x and t is correct, e.g. first element smaller than first (and only positive elements?)."""
        if D <= 0:
            raise ValueError('D must be a positive number.')
        if x_int.ndim != 1:
            raise ValueError('x_int mus be a 1-D numpy array.')
        if t_int.ndim != 1:
            raise ValueError('t_int mus be a 1-D numpy array.')
        if x_int.size > 2:
            raise ValueError('x_int must be of size 2.')
        if t_int.size > 2:
            raise ValueError('t_int must be of size 2.')
        if x_int[1] < x_int[0]:
            raise ValueError('Enter correct interval for x, e.g. second element should be larger than first.')
        if t_int[1] < t_int[0]:
            raise ValueError('Enter correct interval for t, e.g. second element should be larger than first.')
        


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
        """
        self._validate_attributes(equation, M, N)
        
        self.eq = equation
        self.m = M - 1
        self.n = N
        
        if self.h == None:
            self._update_step_size()
            self._check_num_stability()

# --- public methods ---

    def create_solution_matrix(self) -> np.ndarray:
        """Creates the solution matrix, an (M-1) x (N+1) np.array filled with zeros.
        Returns:
            w (np.array): the solution matrix"""
        
        w = np.zeros((self.m, self.n + 1))
        return w
    

    def add_initial_condition(self, w: np.ndarray) -> np.ndarray:
        """Adds the initial condition as a column vector to the solution matrix.
        Parameters:
            w (np.ndarray) : solution matrix without initial values
        Returns:
            w (np.ndarray) : solution matrix with added initial condition"""
        
        if not isinstance(w, np.ndarray):
            raise TypeError('w must be an numpy array.')

        w[:,0] = self._initial_condition()
        return w
    

    def add_boundary_conditions(self, w: np.ndarray) -> np.ndarray:
        """Expands the solution matrix and adds the left and right boundary conditions as first and last column.
        Parameters:
            w (np.ndarray) : solution matrix with solution
        Returns:
            w_comp (np.ndarray) : expanded solution matrix with added boundary conditions"""
        
        if not isinstance(w, np.ndarray):
            raise TypeError('w must be an numpy array.')

        w_comp = np.zeros((self.m + 2, self.n+1))
        w_comp[0, :] = self._boundary_condition(self.eq.l)
        w_comp[1:-1, :] = w
        w_comp[-1, :] = self._boundary_condition(self.eq.r)

        return w_comp
    

    def one_step(self, w: np.ndarray, j: int) -> np.ndarray:
        """Takes one step with the forward difference method. Returns the updated solution matrix.
        Parameters:
            w (np.array): solution matrix to update
            j (int): current column of solution matrix
        Returns:
            updated solution matrix"""
        
        if not isinstance(w, np.ndarray):
            raise TypeError('w must be an numpy array.')
        if not isinstance(j, int):
            raise TypeError('j must be an integer.')

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


# --- private methods ---

    def _validate_attributes(self, equation: HeatEquation, M: int, N: int):
        """Checks the data types of the parameters passed to the class. Raises TypeError if wrong data type."""
        if not isinstance(equation, HeatEquation):
            raise TypeError('equation must be an instance of class HeatEquation.')
        if not isinstance(M, int):
            raise TypeError('M must be an integer.')
        if M <= 0:
            raise ValueError('M must be a positive.')
        if not isinstance(N, int):
            raise TypeError('N must be an integer.')
        if N <= 0:
            raise ValueError('N must be positive.')

    def _update_step_size(self):
        """Updates the step sizes h for space and t for time once when an instance of the class is created."""
        self.h = (self.eq.x_int[1] - self.eq.x_int[0]) / (self.m + 1)
        self.k = (self.eq.t_int[1] - self.eq.t_int[0]) / (self.n)

    def _check_num_stability(self):
        """Checks that the step sizes guarantee a numerically stable solution, using theorem for 
        numerical stability for forward difference method. Displays waring if condition for stability is not met."""
        constant = (self.eq.D * self.k) / (self.h ** 2)
        if constant < (1/2):
            return
        else:
            print('Choose different values for M and/or N to ensure numerical stability.')

    def _initial_condition(self):
        """Creates a numpy array with initial values."""
        x_points = self.eq.x_int[0] + np.arange(1, self.m+1) * self.h
        inital_condition = self.eq.create_init_cond_array(x_points)
        return inital_condition

    def _boundary_condition(self, g: callable):
        """Creates a numpy array with boundary values."""
        t_points = self.eq.t_int[0] + np.arange(self.n+1) * self.k
        boundary_condition = self.eq.create_bound_cond_array(g, t_points)
        return boundary_condition

    def _sigma(self):
        """Calculates the coefficent sigma used in the coefficent matrix."""
        sigma = self.eq.D * self.k / (self.h ** 2)
        return sigma

    def _coefficent_matrix(self):
        """Creates the coefficent matrix, that is later multiplied with the solution matrix 
        in every step in the t direction."""
        sigma = self._sigma()
        main_diagonal = (1 - 2*sigma)*np.ones(self.m)
        super_diagonal = sigma * np.ones(self.m - 1)

        A = np.diag(main_diagonal) + np.diag(super_diagonal, 1) + np.diag(super_diagonal, -1)
        return A