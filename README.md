# Heat equation solver
## API for solving the 1D heat equation
A small API that provides functions for solving the 1D heat equation using forward differences numerical method.
## Installation

## Usage
Once installed, the API can be used to create an instance of the class HeatEquation to represent the equation to be solved. Then the class ForwardDiff can be used to solve the equation numerically with forward differences.
### Example
To solve the equation:
$`\begin{cases}
\frac{\partial u}{\partial t}-\frac{\partial^{2}u}{\partial x^{2}}=0 \\
u(0,t)=0 \\
u(1,t)=0 \\
u(x,0)=2\sin(2\pi t)
\end{cases}`$
first create an instance of the heat equation class:
```python
from heat import *

equation = HeatEquation(
	D=1,
	f=lambda x: 2*np.sin(2*np.pi*x),
	l=lambda t: 0*t,
	r=lambda t: 0*t,
	x_int=[0,1],
	t_int=[0,1]
)
```
Then create an instance of the solution class:
```python
solver = ForwardDiff(
	equation,
	M=10,
	N=100,
)
```
Solve the equation and store the solution in variable solution:
```python
solution = solver.solve()
```

## Dependencies
This API uses Numpy to handle matrix operations. In order to use the API, Numpy must be installed. More about Numpy: [numpy.org](https://numpy.org).


