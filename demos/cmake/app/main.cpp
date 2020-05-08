#include <Optima/Optima.hpp>
using namespace Optima;

int main()
{
    // Solve the following problem:
    // min( (x-1)**2 + (y-1)**2 ) subject to x = y and x,y >= 0

    Dims dims;
    dims.x = 2; // number of variables
    dims.be = 1; // number of linear equality constraints

    Problem problem(dims);
    problem.Ae = Matrix{{ {1.0, -1.0} }};
    problem.be = Vector{{ 0.0 }};
    problem.f = [](VectorConstRef x, ObjectiveResult& res)
    {
        res.f = (x[0] - 1)*(x[0] - 1) + (x[1] - 1)*(x[1] - 1);
        res.g = 2.0 * (x - 1);
        res.H = 2.0 * Matrix::Identity(2, 2);
    };

    State state(dims);

    Options options;
    options.output.active = true;

    Solver solver(problem);
    solver.setOptions(options);

    solver.solve(state, problem);

    std::cout << state.x << std::endl;
}
