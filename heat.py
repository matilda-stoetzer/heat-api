# API:t
import numpy as np
import matplotlib.pyplot as plt


class HeatEquation:
    def __init__(self, D, f, l, r, x_int, t_int):
        self.D = D
        self.f = f
        self.l = l
        self.r = r
        self.x_int = x_int
        self.t_int = t_int

class HeatSolve:

    def __init__(self, equation, M, N):
        self.eq = equation
        self.m = M - 1
        self.n = N
        self.h = None
        self.k = None


    def create_solution_matrix(self):
        """Creates an (m x n+1) np.array filled with zeros, the solution matrix.
        Parameters:
            None
        Returns:
            w (np.array): the solution matrix"""
        w = np.zeros((self.m, self.n + 1))
        return w
    
    def _update_step_size(self):
        self.h = (self.x_int[1] - self.x_int[0]) / (self.m + 1)
        self.k = (self.t_int[1] - self.t_int[0]) / (self.n)

    def _initial_condition(self):
        x_points = self.x_int[0] + np.arange(1, self.m+1) * self.h
        inital_condition = self.f(x_points)
        return inital_condition

    def _boundary_condition(self, g):
        t_points = self.t_int[0] + np.arange(self.n+1) * self.k
        boundary_condition = g(t_points)
        return boundary_condition
    
    def _sigma(self):
        sigma = self.D * self.k / (self.h ** 2)
        return sigma

    def _coefficent_matrix(self): 
        sigma = self._sigma()
        main_diagonal = (1 - 2*sigma)*np.ones(self.m)
        super_diagonal = sigma * np.ones(self.m - 1)

        A = np.diag(main_diagonal) + np.diag(super_diagonal, 1) + np.diag(super_diagonal, -1)
        return A
    
    def one_step(self, w, j):
        """Takes one step with the forward difference method. Returns the updated solution matrix.
        Parameters:
            w (np.array): solution matrix to update
            j (int): current column of solutin matrix
        Returns:
            updated solution matrix"""

        left_side = self._boundary_condition(self.l)
        right_side = self._boundary_condition(self.r)

        v = np.concatenate(([left_side[j]], np.zeros(self.m - 2), [right_side[j]]))
        sigma = self._sigma()
        b = sigma * v

        A = self._coefficent_matrix()
        return A @ w[:, j] + b

    def solve(self):
        """Solves the equation using forward differences. Initializes the solution matrix, 
        sets the initial conditions and updates the solution. Inserts the boundary values.
        Parameters:
            None
        Returns:
            w_comp (n.array): complete solution"""
        w = self.create_solution_matrix()
        self._update_step_size()

        w[:,0] = self._initial_condition()

        for j in range(self.n):
            w[:, j+1] = self.one_step(w, j)

        w_comp = np.zeros((self.m + 2, self.n+1))
        w_comp[0, :] = self._boundary_condition(self.l)
        w_comp[1:-1, :] = w
        w_comp[-1, :] = self._boundary_condition(self.r)
        
        return w_comp

    
def main():
    f = lambda x: np.cosh(x)
    l = lambda t: 2*np.exp(2*t)
    r = lambda t: (np.exp(2) + 1) * np.exp(2*t - 1)
    D = 2
    x_int = np.array([0, 1])
    t_int = np.array([0, 1])

    dx = 0.1
    dt = 0.002

    M = (x_int[1] - x_int[0]) / dx
    N = (t_int[1] - t_int[0]) / dt

    M = np.round(M)
    N = np.round(N)

    M = int(M)
    N = int(N)


    equation = HeatObj(D, f, l, r, M, N, x_int, t_int)
    solution = equation.solve()


    x = np.linspace(x_int[0], x_int[1], equation.m + 2)
    t = np.linspace(t_int[0], t_int[1], equation.n + 1)
    T, X = np.meshgrid(t, x)

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, solution, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    # ax.set_zlim(-1, 1)
    ax.view_init(30, 60)
    plt.show()


if __name__ == '__main__':
    main()

