import torch
import numpy as np
import matplotlib.pyplot as plt


# Ground truth drag function: -0.1 * v * abs(v)
def true_drag(v):
    return -0.1 * v * np.abs(v)


# Training data
v_train = np.linspace(-20, 20, 200).reshape(-1, 1)
drag_train = true_drag(v_train)

# Torch tensors
v_tensor = torch.tensor(v_train, dtype=torch.float32)
drag_tensor = torch.tensor(drag_train, dtype=torch.float32)


class DragModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = DragModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(300):
    pred = model(v_tensor)
    loss = loss_fn(pred, drag_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting the results of the learning
print(f"Final loss: {loss.item():.4f}")
plt.plot(v_train, drag_train, label="Ground truth")
with torch.no_grad():
    plt.plot(v_train, model(v_tensor).numpy(), label="Model")
plt.title("Model output comparison plot")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Drag force (N)")
plt.grid()
plt.show()

import vip_ivp as vip

# Set up the system
mass = 2.0
g = -9.81
h0 = 60  # m

acc = vip.loop_node()
v = vip.integrate(acc, x0=0)
y = vip.integrate(v, x0=h0)

# Compute drag with the neural network
# The model needs a Tensor as an input, so we convert v to a NumPy array, then a tensor
v_np = vip.f(np.atleast_1d)(v)
v_tensor = vip.f(torch.tensor)(v_np, dtype=torch.float32)
# We use our tensor input into the PyTorch model
drag_tensor = vip.f(model)(v_tensor)
# Now we need to convert our drag Tensor to a float
drag = drag_tensor.m(drag_tensor.output_type.item)()

acc.loop_into(g + drag / mass)

# Terminate the simulation on hitting the ground
vip.terminate_on(y.crosses(0, direction="falling"))

# Plotting
y.to_plot("Height (m)")
v.to_plot("Velocity (m/s)")
drag.to_plot("Drag Force (N)")
acc.to_plot("Acceleration (m/s²)")

vip.solve(100, time_step=0.01)

vip.new_system()


def adapt_model(velocity: float, model: torch.nn.Module) -> float:
    v_np = np.atleast_1d(velocity)
    v_tensor = torch.tensor(v_np, dtype=torch.float32)
    result_tensor = model(v_tensor)
    return result_tensor.item()


acc = vip.loop_node()
v = vip.integrate(acc, x0=0)
y = vip.integrate(v, x0=h0)

# Compute drag with the neural network
drag = vip.f(adapt_model)(v, model)

acc.loop_into(g + drag / mass)

# Terminate the simulation on hitting the ground
vip.terminate_on(y.crosses(0, direction="falling"))

# Plotting
y.to_plot("Height (m)")
v.to_plot("Velocity (m/s)")
drag.to_plot("Drag Force (N)")
acc.to_plot("Acceleration (m/s²)")

vip.solve(100, time_step=0.01)
