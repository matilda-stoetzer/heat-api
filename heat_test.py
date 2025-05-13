# tester för heat.py
from heat import *
import pytest

@pytest.fixture(scope='class')
def example_equation():
    return HeatEquation(
        D=1,
        f=lambda x: 2*x,
        l=lambda t: 3*t,
        r=lambda t: 3*t,
        x_int=[0, 1],
        t_int=[0, 1]
    )

@pytest.fixture(scope='class')
def example_solver(example_equation):
    return ForwardDiff(
        equation=example_equation,
        M=10,
        N=100
    )


class TestHe:
    def test_attributes(self, example_equation):
        """Tests that the attributes is the right data type."""
        
        assert callable(example_equation.l)
        assert callable(example_equation.f)

    def test_create_init_cond_array(self, example_equation):
        """Tests that the value passed to the function is the same 
        as if it were passed to the initial condition function itself."""
        
        z = [0, 3]
        assert example_equation.create_init_cond_array(z) == example_equation.f(z)

    def test_create_bound_cond_array(self, example_equation):
        """Tests that the value passed to the function is the same as if it were passed to the 
        boundary condition function itself."""
        
        z = [0, 3]
        assert example_equation.create_bound_cond_array(example_equation.l, z) == example_equation.l(z)
        assert example_equation.create_bound_cond_array(example_equation.r, z) == example_equation.r(z)



class TestForwDiff:
    def test_dimension_sol_matrix(self, example_solver):
        """Tests that the dimension of the solution matrix is (M-1) x (N+1)."""
        
        solution_matrix = example_solver.create_solution_matrix()
        assert solution_matrix.shape == (example_solver.m, example_solver.n+1)

    def test_create_sol_matrix_empty(self, example_solver):
        """Tests that the solution matrix only consists of zeros when first created."""
        
        sol_matrix = example_solver.create_solution_matrix()
        np.testing.assert_array_equal(sol_matrix, np.zeros((9, 101)))


    def test_add_initial_condition(self, example_solver):
        """Tests that the initial conditions is added in the right column."""

        w = example_solver.create_solution_matrix()
        init_cond_vector = example_solver._initial_condition()
        w = example_solver.add_initial_condition(w)

        np.testing.assert_array_equal(w[:,0], init_cond_vector)


    def test_add_boundary_conditions(self, example_solver):
        """Tests that the boundary conditions function updates the dimension of the solution matrix 
        and adds the correct values for the boundary condition."""

        w = example_solver.create_solution_matrix()
        w_comp = example_solver.add_boundary_conditions(w)
        left_bound_val = example_solver._boundary_condition(example_solver.eq.l)
        right_bound_val = example_solver._boundary_condition(example_solver.eq.r)

        assert w.shape != w_comp.shape
        np.testing.assert_array_equal(w_comp[0,:], left_bound_val)
        np.testing.assert_array_equal(w_comp[-1,:], right_bound_val)
        

    def test_one_step(self, example_solver):
        """Tests that the function one_step updates one step."""

        sol_matrix = example_solver.create_solution_matrix()
        left_side = example_solver._boundary_condition(example_solver.eq.l)
        right_side = example_solver._boundary_condition(example_solver.eq.r)
        v = np.concatenate(([left_side[1]], np.zeros(example_solver.m - 2), [right_side[1]]))
        b = example_solver._sigma() * v
        A = example_solver._coefficent_matrix()
        test_matrix = A @ sol_matrix[:, 1] + b

        np.testing.assert_array_equal(example_solver.one_step(sol_matrix, 1), test_matrix)


    def test_solve(self, example_solver):
        """Tests that the solution matrix is not all zero after solve has been used."""

        solution = example_solver.solve()

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(solution, np.zeros((example_solver.m, example_solver.n+1)))
