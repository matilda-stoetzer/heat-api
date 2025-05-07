# tester för main.py
from heat import * # behöver ej importera numpy
import pytest


class TestHeatEq:
    def test_attributes(self):
        """Tests that the attributes is the right type."""
        D = 1
        f = lambda x: 2*x
        l = lambda t: 3*t
        r = l
        x = [0, 1]
        t = [0, 1]

        test_eq = HeatEquation(D, f, l, r, x, t)
        assert type(test_eq.f) == type(f)
        assert type(test_eq.l) == type(l)


    def test_initial_condition(self):
        pass

    def test_boundary_condition(self):
        pass


class TestForwDiff:
    def test_dimension_sol_matrix(self):
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)

        solution_matrix = test_numeric.create_solution_matrix()

        assert solution_matrix.shape == (M-1, N+1)

    def test_create_sol_matrix_empty(self):
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)
        sol_matrix = test_numeric.create_solution_matrix()

        np.testing.assert_array_equal(sol_matrix, np.zeros((9, 101)))

        
    def test_add_initial_condition(self):
        test_eq = HeatEquation(1, lambda x: 2*x, lambda t: 0*t, lambda t: 0*t, [0, 1], [0, 1])
        M = 10
        N = 100
        test_numeric = ForwardDiff(test_eq, M, N)



    def test_add_bondary_conditions(self):
        pass

    def test_solve_update(self):
        pass