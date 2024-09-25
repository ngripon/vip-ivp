import matplotlib.pyplot as plt
import pytest

from main import *


@pytest.fixture
def solver():
    return Solver()

def test_rc_circuit(solver):
    q0_values=np.linspace(1,10,10)
    R_values=np.linspace(1,10,10)
    C_values=np.linspace(1,10,10)

    for q0 in q0_values:
        for R in R_values:
            for C in C_values:
                t=np.linspace(0,100,1001)
                exact_solution=q0*np.exp(-t/(R*C))
                solver.reset()
                dq=solver.loop_node(0)
                q=solver.integrate(dq, q0)
                dq.loop_into(-q/(R*C))
                solver.solve(t[-1],t_eval=t)
                plt.plot(t, exact_solution-q.values)
                # plt.plot(q.t, q.values)
    plt.show()