---
sidebar_position: 6
---

# Events

## What is an Event?

**An event is a condition that triggers an action.**

**Conditions** are boolean Temporal Variables.

**Actions** are triggered at time points when the condition is `True`. They are defined through event creation
functions.

**Events** are created through the following functions and methods. The `trigger` argument is for the condition:

- `terminate_on(trigger)`: Terminate the simulation when triggered.
- `IntegratedVar.reset_on(trigger, new_value)`: Set the Integrated Variable value to `new_value` when triggered.
- `execute_on(trigger, f)`: Execute the function `f` when triggered.

### Example: Bouncing Ball

Consider a bouncing ball:

- **Condition**: The ball hits the ground. We use the `.crossed()` method to create a Temporal Variable that returns
  True when the contact with the ground occurs.
- **Action**: The bounce, which is the reversing of the ball's velocity. We use the `.reset_on(trigger, new_value)`
  method on the velocity to change the value of the velocity when it hits the ground.

## Crossing Trigger Variables

In physical systems, where variables are continuous, you should use **Crossing Trigger Variables for your event
conditions**.

**Crossing Trigger Variables** detects the **exact time** at which a crossing between a Temporal Variable and a value
occurs.

To create a Crossing Trigger Variable, use the `.crosses(value, direction)` method:

```python
# Create the system
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=2)

# highlight-next-line
hit_ground = height.crosses(0, "falling")
```

:::tip
The `value` argument can be a Temporal Variable.
:::

### Crossing Direction

Use the `direction` argument to filter triggers:

- `"rising"`: triggered when the value crosses the threshold from below.
- `"falling"`: triggered when crossing from above.
- `"both"` _(default)_: triggered regardless of crossing direction.

### Temporal Triggers

Two functions are available to create Crossing Trigger Variable from the system clock:

- `timeout_trigger(delay)`: Triggers when `simulation_time==delay`.
- `interval_trigger(delay)`: Triggers regularly at every interval of `delay`.

The `delay` argument is a duration in seconds.

### Conditional Triggers

Use logical operators on the Temporal Variables you use as a trigger to create a conditional trigger variable:

```python
# highlight-next-line
v_min = 0.5  # Minimum velocity for the ball to bounce when hitting the ground

# Create the system
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=2)

hit_ground = height.crosses(0, "falling")
# highlight-next-line
trigger_stop = hit_ground & (abs(velocity) <= v_min)
```

![Conditional bounce](./images/conditional_action.png)

## Creating Events

### Terminal Events

Use `terminate_on(trigger)` to terminate the simulation when a triggers occurs:

```python
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=10)

hit_ground = height.crosses(0, "falling")
# highlight-next-line
vip.terminate_on(hit_ground)

height.to_plot()
vip.solve(10, time_step=0.01)
```

![Terminal event](./images/terminal_event.png)

### Reset an Integrated Variable

Use the `.reset_on(trigger, value)` of an Integrated Variable to instantly change its state:

```python
a = vip.temporal(-9.81)
v = vip.integrate(a, 0)
y = vip.integrate(v, 10)

hit_ground = y.crosses(0, "falling")
# highlight-next-line
v.reset_on(hit_ground, -v)  # Reverse velocity
```

### Custom Events

You can trigger any function with `execute_on(trigger, f)`. The function `f` must be a function with **side-effects**.

```python
vip.execute_on(hit_ground, lambda: print("Hello world"))
```

**Output:**

```
Hello world
```

To access the event time, add an argument to the input function:

```python
def log_time(t):
    print(f"Collision at {t}.")


# ...
vip.execute_on(hit_ground, log_time)
```

**Output:**

```
Collision at 1.4278431229270645.
```


