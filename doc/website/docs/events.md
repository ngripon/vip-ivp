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

## Creating Events

### Terminal Events

Set `terminal=True` to stop the simulation when the event triggers.

```python
acceleration = vip.temporal(-9.81)
velocity = vip.integrate(acceleration, x0=0)
height = vip.integrate(velocity, x0=10)

hit_ground = height.on_crossing(0, terminal=True)

height.to_plot()
vip.solve(10, time_step=0.01)
```

![Terminal event](./images/terminal_event.png)

## Time-Based Events

Time-based events are crossing events applied to simulation time.

### Timeout

Use `vip.set_timeout(action, delay)` to apply an action after a given delay.

### Interval

Use `vip.set_interval(action, delay)` to repeat an action periodically.

## Actions

**Actions are side-effect-only functions executed when events trigger.**

There are two types:

1. **System actions**: modify the simulation state
2. **Custom actions**: run arbitrary code (e.g., logging), but **must not modify the simulation**

### Custom Actions

Define a function with side effects:

```python
height.on_crossing(0, action=lambda: print("Hello"))
height.on_crossing(-1, action=lambda: print("world"))
```

**Output:**

```
Hello
world
```

To access the event time, add an argument to the input function:

```python
def log_time(t):
    print(f"Collision at {t}.")


height.on_crossing(0, action=log_time)
```

### System Actions

System actions alter simulation variables. They can only be used with `TemporalVar` or `IntegratedVar`.

#### Resetting an Integrated Variable

Use `.action_reset_to(value)` to instantly change the state:

```python
velocity.action_reset_to(-velocity)  # Reverse velocity
```

#### Changing a Temporal Variable

Use `.action_set_to(new_value)` for non-integrated `TemporalVar`s:

```python
count.action_set_to(count + 1)  # Increment a counter
```

#### Terminate the Simulation

Use `vip.action_terminate` to stop the simulation from within an action.

#### Disable Events

Access `.action_disable` on an event to prevent it from triggering again.

```python
disable_action = my_event.action_disable
```

### Combining Actions

Use `+` to combine multiple actions into a single one. The action on the **left** side of the operator is executed *
*first**.

```python
action_combo = action1 + action2  # Executes action1, then action2
```

### Conditional Actions

Use `vip.where(condition, action_if_true, action_if_false)`:

```python
bounce = velocity.action_reset_to(-k * velocity)
stop = acceleration.action_set_to(0) + velocity.action_reset_to(0)

conditional = vip.where(abs(velocity) > v_min, bounce, stop)
height.on_crossing(0, conditional)
```

![Conditional bounce](./images/conditional_action.png)
