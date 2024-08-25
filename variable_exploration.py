import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

from main import solver


# The parametrized function to be plotted
v0 = 2
x0 = 5
pos, vit, acc = solver.create_variables((x0, v0))
def f(k, c):
    t_final = 50
    m = 5
    acc.set_value(1 / m * (-c * vit - k * pos))
    solver.solve(t_final)
    return pos.t, pos.values

# Define initial parameters
init_k = 1
init_c=2
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
t,y=f(init_k, init_c)
line, = ax.plot(t, y, lw=2)
ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axk = fig.add_axes([0.25, 0.1, 0.65, 0.03])
k_slider = Slider(
    ax=axk,
    label='k',
    valmin=0.1,
    valmax=30,
    valinit=init_k,
)
axc = fig.add_axes([0.25, 0.05, 0.65, 0.03])

c_slider = Slider(
    ax=axc,
    label='c',
    valmin=0.1,
    valmax=30,
    valinit=init_c,
)

# The function to be called anytime a slider's value changes
def update(val):
    t,y=f(k_slider.val, c_slider.val)
    line.set_data(t,y)
    fig.canvas.draw_idle()
    ax.relim()
    ax.autoscale_view(True, True, True)

# register the update function with each slider
k_slider.on_changed(update)
c_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    k_slider.reset()
button.on_clicked(reset)

plt.show()