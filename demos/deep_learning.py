import torch
import numpy as np
import matplotlib.pyplot as plt
import src.vip_ivp as vip


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
plt.plot(v_train, drag_train)
with torch.no_grad():
    plt.plot(v_train, model(v_tensor).numpy())
plt.grid()
plt.show()

# Set up the system
mass = 1.0
g = -9.81

acc = vip.loop_node()
v = vip.integrate(acc, x0=0)
y = vip.integrate(v, x0=50)
# Compute drag with the neural network
v_np = vip.f(np.atleast_1d)(v)
v_tensor = vip.f(torch.tensor)(v_np, dtype=torch.float32)
drag_tensor = vip.f(model)(v_tensor)
drag = vip.f(float)(drag_tensor)

acc.loop_into(g + drag / mass)

# Terminate the simulation on hitting the ground
y.on_crossing(0, direction="falling", terminal=True)

# Plotting
v.to_plot()
drag.to_plot()
y.to_plot()

vip.solve(100, time_step=0.01, verbose=True)
