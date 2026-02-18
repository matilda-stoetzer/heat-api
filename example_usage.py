# Example usage of the API

# import the classes
from heat import *

# create the equation
example_eq = HeatEquation(
    D=1,
    f=lambda x: np.sin(2*np.pi*x)**2,
    l=lambda t: 0*t,
    r=lambda t: 0*t,
    x_int=np.array([0,1]),
    t_int=np.array([0,0.5])
)

# create the solver
solver = ForwardDiff(
    equation=example_eq,
    M=10,
    N=100
)

# solve and store the solution
solution = solver.solve()

# plot the solution using matplotlib.pyplot (optional)
import matplotlib.pyplot as plt

x = np.linspace(example_eq.x_int[0], example_eq.x_int[1], solver.m + 2)
t = np.linspace(example_eq.t_int[0], example_eq.t_int[1], solver.n + 1)
T, X = np.meshgrid(t, x)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, solution, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlim(-0.5, 1)
ax.view_init(30, 60)
plt.show()