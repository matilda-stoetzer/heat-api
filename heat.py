# API:t
import numpy as np
import matplotlib.pyplot as plt

def create_solution_matrix(M, N):
    """Initializes the solution matrix.
    Parameters:
        M (int): number of steps in the x direction
        N (int): number of steps in the t direction
    Returns:
        w (np.array): (m x n+1)-matrix with zeros
    """
    m = M - 1
    n = N

    w = np.zeros((m, n+1))
    return w

def _coefficent_matrix(x_int, t_int, M, N, D):
    """Creates the coefficent matrix for the forward difference method.
    Parameters:
        x_int (np.array): x interval
        t_int (np.array): t interval
        M (int): number of steps in the x direction
        N (int): number of steps in the t direction
        D (int): diffusion constant
    Returns:
        A (np.array): the coefficent matrix
    """
    # konstanter
    h = x_int[1] - x_int[0] / M
    k = t_int[1] - t_int[0] / N
    m = M - 1
    n = N
    sigma = D * k / (h**2)

    # matrisen
    main_diagonal = (1 - 2*sigma)*np.ones(m)
    super_diagonal = sigma*np.ones(m-1)
    A = np.diag(main_diagonal) + np.diag(super_diagonal, 1) + np.diag(super_diagonal, -1)
    return A

def _boundary_values(left_bond, right_bond, t_int, M, N, j):
    """Calculates the boundary values of the solution given the boundary conditions as an np.array with zeros in the middle.
    Parameters:
        None
    Returns:
        left_side (np.array): left side boundary value array
        right_side (np.array): right side boundary value array
    """
    m = M - 1
    n = N
    k = t_int[1] - t_int[0] / N

    left_side = left_bond(t_int[0] + np.arange(n+1)*k)
    right_side = right_bond(t_int[0] + np.arange(n+1)*k)

    return np.concatenate(([left_side[j]], np.zeros(m-2), [right_side[j]]))

def _initial_condition(f, w, x_int, M):
    """Calculates the inital value for all space points x and adds it to the solution matrix.
    Parameters:
        f (callable): initial condition
        w (np.array): solution matrix
        x_int (np.array): x interval
        M (int):
    Returns:
        w (np.array): solution matrix"""
    m = M - 1
    h = x_int[1] - x_int[0] / M
    
    w[:,0] = f(x_int[0] + np.arange(1, m+1) * h)

    return w

def one_step(w, j, t_int, x_int, M, N, D, left_bond, right_bond):
    """Takes one step in the t-direction.
    Parameters:
        w (np.array): part of the solution matrix
        j (int): column to step to
    Returns:
        w_updated (np.array): updated solution matrix
    """
    h = x_int[1] - x_int[0] / M
    k = t_int[1] - t_int[0] / N
    sigma = D * k / (h**2)
    A = _coefficent_matrix(t_int, x_int, M, N, D)
    b = sigma * _boundary_values(left_bond, right_bond, t_int, M, N, j)
    w[:, j+1] = A @ w[:,j] + b
    return w

def solve(t_int, x_int, M, N, D, left_bond, right_bond, f):
    """Solves the equation by stepping forward in time.
    Parameters:
        None
    Returns:
        w_comp (np.array): the completed solution
    """
    m = M - 1
    n = N
    k = t_int[1] - t_int[0] / N

    w = create_solution_matrix(M, N)

    w = _initial_condition(f, w, x_int, M)
    for j in range(n):
        w = one_step(w, j, t_int, x_int, M, N, D, left_bond, right_bond)

    left_side = left_bond(t_int[0] + np.arange(n+1)*k)
    right_side = right_bond(t_int[0] + np.arange(n+1)*k)

    w_full = np.zeros((m + 2, n + 1))
    w_full[0, :] = left_side
    w_full[1:-1, :] = w
    w_full[-1, :] = right_side
    return w_full



def main():

    M = 10
    N = 100
    x = [0, 1]
    t = [0, 0.5]
    f = lambda x: np.sin(2*np.pi*x)**2
    r = lambda t: 0*t
    l = lambda t: 0*t
    W = solve(t, x, M, N, 1, l, r, f)

    m = M - 1
    n = N
    # Meshgrid for plotting
    x = np.linspace(x[0], x[1], m + 2)
    t = np.linspace(t[0], t[1], n + 1)
    T, X = np.meshgrid(t, x)

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, W, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlim(-1, 1)
    ax.view_init(30, 60)
    plt.show()



if __name__ == '__main__':
    main()