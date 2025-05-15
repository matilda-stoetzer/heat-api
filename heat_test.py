from heat import *
import pytest

@pytest.fixture(scope='class')
def example_equation():
    return HeatEquation(
        D=1,
        f=lambda x: 2*x,
        l=lambda t: 3*t,
        r=lambda t: 3*t,
        x_int=np.array([0, 1]),
        t_int=np.array([0, 1])
    )

@pytest.fixture(scope='class')
def example_solver(example_equation: HeatEquation):
    return ForwardDiff(
        equation=example_equation,
        M=10,
        N=100
    )


@pytest.fixture(scope='class')
def wrong_equation(request):
    return HeatEquation(
        D=1,
        f=lambda x: 2*x,
        l=lambda t: 3*t,
        r=lambda t: 3*t,
        x_int=request.param,
        t_int=np.array([0, 1])
    )


class TestHeatEq:
    def test_attributes(self, example_equation: HeatEquation):
        """Tests that the attributes is the right data type."""
        
        assert callable(example_equation.l)
        assert callable(example_equation.f)

    def test_create_init_cond_array(self, example_equation: HeatEquation):
        """Tests that the value passed to the function is the same 
        as if it were passed to the initial condition function itself."""
        
        z = np.array([0, 3])
        assert np.all(example_equation.create_init_cond_array(z)) == np.all(example_equation.f(z))

    def test_create_bound_cond_array(self, example_equation: HeatEquation):
        """Tests that the value passed to the function is the same as if it were passed to the 
        boundary condition function itself."""
        
        z = np.array([0, 3])
        assert np.all(example_equation.create_bound_cond_array(example_equation.l, z)) == np.all(example_equation.l(z))
        assert np.all(example_equation.create_bound_cond_array(example_equation.r, z)) == np.all(example_equation.r(z))

    
    def test_wrong_value_D(self):
        """Tests that ValueError is raised when wrong values for the constant D is entered."""

        with pytest.raises(ValueError):
            test_eq = HeatEquation(
                    D=-1,
                    f=lambda x: 2*x,
                    l=lambda t: 3*t,
                    r=lambda t: 3*t,
                    x_int=np.array([0, 1]),
                    t_int=np.array([0, 1])
                )
            test_eq2 = HeatEquation(
                    D=0,
                    f=lambda x: 2*x,
                    l=lambda t: 3*t,
                    r=lambda t: 3*t,
                    x_int=np.array([0, 1]),
                    t_int=np.array([0, 1])
                )

    def test_wrong_interval(self):
        """Tests that wrong input for intervals raises ValueError."""

        with pytest.raises(ValueError):

            test_eq = HeatEquation(
                D=1,
                f=lambda x: 2*x,
                l=lambda t: 3*t,
                r=lambda t: 3*t,
                x_int=np.array([1, 0]),
                t_int=np.array([0, 1])
            )
            test_eq2 = HeatEquation(
                D=1,
                f=lambda x: 2*x,
                l=lambda t: 3*t,
                r=lambda t: 3*t,
                x_int=np.array([0, 1, 2]),
                t_int=np.array([0, 1])
            )


class TestForwDiff:
    def test_wrong_value_MN(self, example_equation):
        """Tests that ValueError is raised when wrong values for M and/or N are entered."""

        with pytest.raises(ValueError):
            solver = ForwardDiff(example_equation, -10, 100)
            solver2 = ForwardDiff(example_equation, 9, -100)
            solver3 = ForwardDiff(example_equation, -10, 0)


    def test_check_numer_stability(self, capsys, example_equation):
        """Tests that a warning is printed if the selected values for M and/or N makes the solution numerically unstable."""
        solver = ForwardDiff(example_equation, 10, 50)
        printed = capsys.readouterr()
        assert printed.out == 'Choose different values for M and/or N to ensure numerical stability.\n'


    def test_dim_sol_matrix_correct(self, example_solver: ForwardDiff):
        """Tests that the dimension of the solution matrix is (M-1) x (N+1)."""
        
        solution_matrix = example_solver.create_solution_matrix()
        assert solution_matrix.shape == (example_solver.m, example_solver.n+1)

    def test_create_sol_matrix_empty(self, example_solver: ForwardDiff):
        """Tests that the solution matrix only consists of zeros when first created."""
        
        sol_matrix = example_solver.create_solution_matrix()
        np.testing.assert_array_equal(sol_matrix, np.zeros((9, 101)))


    def test_add_initial_condition(self, example_solver: ForwardDiff):
        """Tests that the initial conditions is added in the right column."""

        w = example_solver.create_solution_matrix()
        init_cond_vector = example_solver._initial_condition()
        w = example_solver.add_initial_condition(w)

        np.testing.assert_array_equal(w[:,0], init_cond_vector)


    def test_add_boundary_conditions(self, example_solver: ForwardDiff):
        """Tests that the boundary conditions function updates the dimension of the solution matrix 
        and adds the correct values for the boundary condition."""

        w = example_solver.create_solution_matrix()
        w_comp = example_solver.add_boundary_conditions(w)
        left_bound_val = example_solver._boundary_condition(example_solver.eq.l)
        right_bound_val = example_solver._boundary_condition(example_solver.eq.r)

        assert w.shape != w_comp.shape
        np.testing.assert_array_equal(w_comp[0,:], left_bound_val)
        np.testing.assert_array_equal(w_comp[-1,:], right_bound_val)
        

    def test_one_step(self, example_solver: ForwardDiff):
        """Tests that the function one_step updates one step."""

        sol_matrix = example_solver.create_solution_matrix()
        left_side = example_solver._boundary_condition(example_solver.eq.l)
        right_side = example_solver._boundary_condition(example_solver.eq.r)
        v = np.concatenate(([left_side[1]], np.zeros(example_solver.m - 2), [right_side[1]]))
        b = example_solver._sigma() * v
        A = example_solver._coefficent_matrix()
        test_matrix = A @ sol_matrix[:, 1] + b

        np.testing.assert_array_equal(example_solver.one_step(sol_matrix, 1), test_matrix)


    def test_solve(self, example_solver: ForwardDiff):
        """Tests that the solution matrix is not all zero after solve has been used."""

        solution = example_solver.solve()

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(solution, np.zeros((example_solver.m, example_solver.n+1)))