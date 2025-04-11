import torch
import numpy as np
import matplotlib.pyplot as plt
import src.vip_ivp as vip


# Ground truth drag function: -0.1 * v * abs(v)
def true_drag(v):
    return -0.1 * v * np.abs(v)

def torch_model_as_numpy_fn(model: torch.nn.Module):
    def wrapped(x):
        x_np = np.atleast_1d(x)
        is_scalar = np.isscalar(x) or x_np.ndim == 0

        # Ensure input is 2D (batch_size, in_features)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)

        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            y_tensor = model(x_tensor)
        y = y_tensor.numpy()

        if is_scalar:
            return y.flat[0]
        elif y.shape[-1] == 1:
            return y[...,0]  # Return a 1D array
        return y  # Return a 2D array
    return wrapped


# Training data
v_train = np.linspace(-100, 100, 200).reshape(-1, 1)
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

print(f"Final loss: {loss.item():.4f}")

plt.plot(v_train, drag_train)
np_fun=torch_model_as_numpy_fn(model)
plt.plot(v_train[0], np_fun(v_train[0]))
plt.grid()
plt.show()

print(np_fun(np.linspace(0,10,11)))
print(np_fun(5))
print(np_fun(5).shape)

# Set up the system
mass = 1.0
g = -9.81

acc = vip.loop_node()
v = vip.integrate(acc, x0=0)  # Start with upward speed
y = vip.integrate(v, x0=10)

drag = vip.f(torch_model_as_numpy_fn(model))(v)

acc.loop_into(g + drag / mass)

v.to_plot()
drag.to_plot()
y.to_plot()

vip.solve(5, time_step=0.01)
