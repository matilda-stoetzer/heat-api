# tester för heat.py
from heat import *
import pytest


class TestHeatEq:
    def test_attributes(self):
        """Tests that the attributes is the right data type."""
        D = 1
        f = lambda x: 2*x
        l = lambda t: 3*t
        r = l
        x = [0, 1]
        t = [0, 1]
        test_eq = HeatEquation(D, f, l, r, x, t)

        assert type(test_eq.f) == type(f)
        assert type(test_eq.l) == type(l)

    def test_create_init_cond_array(self):
        """Tests that the value passed to the function is the same as if it were passed to the initial condition function itself."""
        D = 1
        f = lambda x: 2*x
        l = lambda t: 3*t
        r = l
        x = [0, 1]
        t = [0, 1]
        test_eq = HeatEquation(D, f, l, r, x, t)
        z = [0, 3]

        assert test_eq.create_init_cond_array(z) == f(z)

    def test_create_bound_cond_array(self):
        """Tests that the value passed to the function is the same as if it were passed to the boundary condition function itself."""
        D = 1
        f = lambda x: 2*x
        l = lambda t: 3*t
        r = l
        x = [0, 1]
        t = [0, 1]
        test_eq = HeatEquation(D, f, l, r, x, t)
        z = [0, 3]
        
        assert test_eq.create_bound_cond_array(l, z) == l(z)
        assert test_eq.create_bound_cond_array(r, z) == r(z)


class TestForwDiff:
    def test_dimension_sol_matrix(self):
        """Tests that the dimension of the solution matrix is (M-1) x (N+1)."""
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)
        solution_matrix = test_numeric.create_solution_matrix()

        assert solution_matrix.shape == (M-1, N+1)

    def test_create_sol_matrix_empty(self):
        """Tests that the solution matrix only consists of zeros when first created."""
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)
        sol_matrix = test_numeric.create_solution_matrix()

        np.testing.assert_array_equal(sol_matrix, np.zeros((9, 101)))


    def test_add_initial_condition(self):
        """Tests that the initial conditions is added in the right column."""
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)
        w = test_numeric.create_solution_matrix()
        init_cond_vector = test_numeric._initial_condition()

        w = test_numeric.add_initial_condition(w)

        np.testing.assert_array_equal(w[:,0], init_cond_vector)


    def test_add_boundary_conditions(self):
        """Tests that the boundary conditions function updates the dimension of the solution matrix 
        and adds the correct values for the boundary condition."""
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)

        w = test_numeric.create_solution_matrix()
        w_comp = test_numeric.add_boundary_conditions(w)
        left_bound_val = test_numeric._boundary_condition(test_numeric.eq.l)
        right_bound_val = test_numeric._boundary_condition(test_numeric.eq.r)

        assert w.shape != w_comp.shape
        np.testing.assert_array_equal(w_comp[0,:], left_bound_val)
        np.testing.assert_array_equal(w_comp[-1,:], right_bound_val)
        

    def test_one_step(self):
        """Tests that the function one_step updates one step."""
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)

        sol_matrix = test_numeric.create_solution_matrix()


        left_side = test_numeric._boundary_condition(test_numeric.eq.l)
        right_side = test_numeric._boundary_condition(test_numeric.eq.r)
        v = np.concatenate(([left_side[1]], np.zeros(test_numeric.m - 2), [right_side[1]]))
        b = test_numeric._sigma() * v
        A = test_numeric._coefficent_matrix()
        test_matrix = A @ sol_matrix[:, 1] + b

        np.testing.assert_array_equal(test_numeric.one_step(sol_matrix, 1), test_matrix)


    def test_solve(self):
        """Tests that the solution matrix is not all zero after solve has been used."""

        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)

        solution = test_numeric.solve()

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(solution, np.zeros((M+1, N+1)))
