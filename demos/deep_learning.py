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

def adapt_model(velocity: float, model: torch.nn.Module) -> float:
    v_np = np.atleast_1d(velocity)
    v_tensor = torch.tensor(v_np, dtype=torch.float32)
    result_tensor = model(v_tensor)
    return result_tensor.item()


y, v = vip.n_order_state(h0, 0)

# Compute drag with the neural network
drag = vip.f(adapt_model)(v, model)

v.der=g + drag / mass

# Terminate the simulation on hitting the ground
vip.when(y.crosses(0, direction="falling"), vip.terminate)


vip.solve(100, step_eval=0.01)

# Plotting
vip.plot(y, v, drag, v.der)
