import vip_ivp as vip
import matplotlib.pyplot as plt


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        print(f"Counter incremented to {self.count}")


counter = Counter()

# System parameters
m = 1.0  # Mass (kg)
k = 10.0  # Spring constant (N/m)
c = 0.5  # Damping coefficient (N·s/m)

x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)

# Build the system
x, v = vip.n_order_state(x0, v0)
v.der = -(c * v + k * x) / m  # Set acceleration value

# Create event that triggers when x crosses 0
zero_crossing = x.crosses(0)
vip.when(zero_crossing, counter.increment)

# Solve the system
vip.solve(10)

# Use the plot function
vip.plot(x)

# Create a phase space diagram
plt.plot(x.values, v.values)
plt.xlabel('Position x (m)')
plt.ylabel('Velocity v (m/s)')
plt.title('Phase Space Diagram')
plt.grid()
plt.show()

# Export the results to pandas
dataframe = vip.export_to_df(v, x)
print(dataframe)
