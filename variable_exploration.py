import inspect
from inspect import signature

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

from main import Solver

def explore(f):
    params=signature(f).parameters
    init_params=[param.default if param.default is not inspect.Parameter.empty else 1 for param in params.values()]
    print(init_params)
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    t,y=f(*init_params)
    line, = ax.plot(t, y, lw=2)
    ax.set_xlabel('Time [s]')

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    sliders=[]
    for i,param in enumerate(params.keys()):
        slider_ax = fig.add_axes([0.25, 0.05+0.05*i, 0.65, 0.03])
        slider = Slider(
            ax=slider_ax,
            label=param,
            valmin=0.1,
            valmax=30,
            valinit=init_params[i],
        )
        sliders.append(slider)



    # The function to be called anytime a slider's value changes
    def update(val):
        t,y=f(*(slider.val for slider in sliders))
        line.set_data(t,y)
        fig.canvas.draw_idle()
        ax.relim()
        ax.autoscale_view(True, True, True)

    # register the update function with each slider
    [slider.on_changed(update) for slider in sliders]


    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        [slider.reset() for slider in sliders]
    button.on_clicked(reset)

    plt.show()

if __name__ == '__main__':
    # The parametrized function to be plotted
    solver = Solver()
    v0 = 2
    x0 = 5
    pos, vit, acc = solver.create_variables((x0, v0))


    def f(k=2, c=3, m=5):
        t_final = 50
        acc.set_value(1 / m * (-c * vit - k * pos))
        solver.solve(t_final)
        return pos.t, pos.values

    explore(f)
