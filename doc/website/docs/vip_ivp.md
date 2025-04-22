# API Reference

## Functions

### clear()

Clear the current solver’s stored information.

* **Return type:**
  `None`

### create_scenario(scenario_table, time_key, interpolation_kind='linear', sep=',')

Creates a scenario from a given input table, which can be in various formats such as CSV, JSON, dictionary, or DataFrame.
The maps in the scenario table are interpolated over time and converted into TemporalVar objects.
The function processes the data and returns a dictionary of TemporalVar objects.

* **Parameters:**
  * **scenario_table** (*Union* *[**pd.DataFrame* *,* *str* *,* *dict* *]*) – The input data, which can be one of the following formats:
    - A CSV file path (string)
    - A JSON file path (string)
    - A dictionary of data
    - A pandas DataFrame
  * **time_key** (*str*) – The key (column) to use as time for the scenario.
  * **interpolation_kind** – Specifies the kind of interpolation as a string or as an integer specifying the order of

the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next
value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that
‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
:type interpolation_kind: str or int, optional

* **Parameters:**
  **sep** (*str* *,* *optional*) – The separator to use when reading CSV files. Default is a comma.
* **Returns:**
  A dictionary of TemporalVar objects representing the scenario, where the keys are the variables and the
  values are the corresponding TemporalVar instances.
* **Return type:**
  Dict[Any, TemporalVar]
* **Raises:**
  **ValueError** – If the input file type is unsupported or the input type is invalid.

### explore(fun, t_end, bounds=(), time_step=None, title='')

Explore the function f over the given bounds and solve the system until t_end.
This function needs the sliderplot package.

* **Parameters:**
  * **title** (`str`) – Title of the plot
  * **time_step** (`float`) – Time step of the simulation
  * **fun** (`Callable`[`...`, `TypeVar`(`T`)]) – The function to explore.
  * **t_end** (`float`) – Time at which the integration stops.
  * **bounds** – Bounds for the exploration.
* **Return type:**
  `None`

### export_file(filename, variable_list=None, file_format=None)

* **Return type:**
  `None`

### export_to_df(\*variables)

* **Return type:**
  pd.DataFrame

### f(func)

* **Return type:**
  `Callable`[[`ParamSpec`(`P`)], `TemporalVar`[`TypeVar`(`T`)]]

### get_events()

* **Return type:**
  `List`[`Event`]

### get_time_variable()

Return the time variable of the system.
:rtype: `TemporalVar`[`float`]
:return: Time variable that returns the time value of each simulation step.

### get_var(var_name)

Retrieve a saved TemporalVar by its name.

* **Parameters:**
  **var_name** (`str`) – The name of the saved TemporalVar.
* **Return type:**
  `TemporalVar`
* **Returns:**
  The retrieved TemporalVar.

### integrate(input_value, x0, minimum=None, maximum=None)

Integrate the input value starting from the initial condition x0.

* **Parameters:**
  * **minimum** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`), `None`]) – Lower integration bound. Can be a TemporalVar
  * **maximum** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`), `None`]) – Higher integration bound. Can be a TemporalVar
  * **input_value** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]) – The value to be integrated, can be a TemporalVar or a number.
  * **x0** (`Union`[`TypeVar`(`T`), `List`]) – The initial condition for the integration.
* **Return type:**
  `IntegratedVar`[`TypeVar`(`T`)]
* **Returns:**
  The integrated TemporalVar.

### loop_node(shape=None, strict=True)

Create a loop node. Loop node can accept new inputs through its “loop_into()” method after being instantiated.

* **Parameters:**
  * **shape** (`Union`[`int`, `tuple`[`int`, `...`]]) – Shape of the NumPy array contained in the Loop Node. If None, the Loop Node will contain a scalar.
  * **strict** (`bool`) – Flag that triggers an error when the Loop Node has not been set before the solving.
* **Return type:**
  `LoopNode`
* **Returns:**
  The created LoopNode.

### new_system()

Create a new solver system.

* **Return type:**
  `None`

### plot()

Plot the variables that have been marked for plotting.

* **Return type:**
  `None`

### save(\*args)

Save the given TemporalVars with their variable names.

* **Parameters:**
  **args** (`TemporalVar`) – TemporalVars to be saved.
* **Raises:**
  **ValueError** – If any of the arguments is not a TemporalVar.
* **Return type:**
  `None`

### set_interval(action, delay)

* **Return type:**
  `Event`

### set_timeout(action, delay)

* **Return type:**
  `Event`

### solve(t_end, time_step=0.1, method='RK45', t_eval=None, include_events_times=True, plot=True, rtol=0.001, atol=1e-06, max_step=inf, verbose=False)

Solve the equations of the dynamical system through a hybrid solver.

The hybrid solver is a modified version of SciPy’s solve_ivp() function.

* **Parameters:**
  * **max_step** – Maximum allowed step size. Default is np.inf, i.e., the step size is not bounded and determined
    solely by the solver.
  * **plot** (`bool`) – If True, a plot will show the result of the simulation for variables that were registered to plot.
  * **verbose** (`bool`) – If True, print solving information to the console.
  * **include_events_times** (`bool`) – If True, include time points at which events are triggered.
  * **t_end** (`float`) – Time at which the integration stops.
  * **method** – Integration method to use. Default is ‘RK45’. For a list of available methods, see SciPy’s
    solve_ivp() documentation.
  * **time_step** (`Optional`[`float`]) – Time step for the integration. If None, use points selected by the solver.
  * **t_eval** (`Union`[`List`, `ndarray`[`tuple`[`int`, `...`], `dtype`[`TypeVar`(`_ScalarType_co`, bound= `generic`, covariant=True)]]]) – Times at which to store the computed solution. If None, use points selected by the solver.
  * **rtol** (`float`) – Relative tolerance. The solver keeps the local error estimates less than atol + rtol \* abs(y).
  * **atol** (`float`) – Absolute tolerance. The solver keeps the local error estimates less than atol + rtol \* abs(y).
* **Return type:**
  `None`

### temporal(value)

Create a Temporal Variable from a temporal function, a scalar value, a dict, a list or a NumPy array.
If the input value is a list, the variable content will be converted to a NumPy array. As a consequence, a nested
list must represent a valid rectangular matrix.

* **Parameters:**
  **value** (`Union`[`Callable`[[`ndarray`[`tuple`[`int`, `...`], `dtype`[`TypeVar`(`_ScalarType_co`, bound= `generic`, covariant=True)]]], `TypeVar`(`T`)], `TypeVar`(`T`)]) – A function f(t), a scalar value, a dict, a list or a NumPy array.
* **Return type:**
  `TemporalVar`[`TypeVar`(`T`)]
* **Returns:**
  The created TemporalVar.

### where(condition, a, b)

## Classes

### *class* Action(fun, expression=None)

Bases: `object`

#### \_\_init_\_(fun, expression=None)

### *class* Event(solver, fun, action, direction='both', terminal=False)

Bases: `object`

#### \_\_init_\_(solver, fun, action, direction='both', terminal=False)

#### *property* action_disable *: Action*

Create an action that disable the event, so it will not execute its action anymore.
:return: Action

#### clear()

#### evaluate(t, y)

* **Return type:**
  `None`

#### execute_action(t, y)

### *class* IntegratedVar(solver, fun=None, expression=None, x0=None, minimum=-inf, maximum=inf, y_idx=None)

Bases: `TemporalVar`[`T`]

#### \_\_init_\_(solver, fun=None, expression=None, x0=None, minimum=-inf, maximum=inf, y_idx=None)

#### action_reset_to(value)

Create an action that, when its event is triggered, reset the IntegratedVar output to the specified value.
:type value: `Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]
:param value: Value at which the integrator output is reset to
:rtype: `Action`
:return: Action to be put into an Event.

#### action_set_to()

Not implemented
:rtype: `None`
:return: Raise an exception

#### *property* y_idx

### *class* LoopNode(solver, shape=None, strict=True)

Bases: `TemporalVar`[`T`]

#### \_\_init_\_(solver, shape=None, strict=True)

#### action_set_to()

Not implemented
:rtype: `None`
:return: Raise an exception

#### is_valid()

Check if the Loop Node is ready to be solved.
If the Loop Node uses strict mode, its value must be set.
:rtype: `bool`
:return: True if valid, False if incorrect

#### loop_into(value, force=False)

Set the input value of the loop node.

* **Parameters:**
  * **force** (`bool`) – Add the value to the loop node even if it has already been set.
  * **value** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`), `List`]) – The value to add, can be a TemporalVar or a number.

### *class* TemporalVar(solver, source=None, expression=None, child_cls=None, operator=None, call_mode=CallMode.CALL_ARGS_FUN, is_discrete=False)

Bases: `Generic`[`T`]

#### \_\_init_\_(solver, source=None, expression=None, child_cls=None, operator=None, call_mode=CallMode.CALL_ARGS_FUN, is_discrete=False)

#### action_set_to(new_value)

Create an action that, when its event is triggered, change the TemporalVar value.
This action works with recursive statements. For example, count.action_set_to(count+1) is valid.
:type new_value: `Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]
:param new_value: New value
:rtype: `Action`
:return: Action to be put into an Event.

#### clear()

#### delayed(delay, initial_value=0)

Create a delayed version of the TemporalVar.
:type delay: `int`
:param delay: Number of solver steps by which the new TemporalVar is delayed.
:type initial_value: `TypeVar`(`T`)
:param initial_value: Value of the delayed variable at the beginning when there is not any value for the original value.
:rtype: `TemporalVar`[`TypeVar`(`T`)]
:return: Delayed version of the TemporalVar

#### derivative(initial_value=0)

Return the derivative of the Temporal Variable.

Warning: Contrary to integration, this derivative method does not guarantee precision. Use it only as an escape
hatch.
:type initial_value: 
:param initial_value: value at t=0
:rtype: `TemporalVar`[`TypeVar`(`T`)]
:return: TemporalVar containing the derivative.

#### *property* expression

#### *classmethod* from_scenario(solver, scenario_table, time_key, interpolation_kind='linear')

* **Return type:**
  `TemporalVar`

#### items()

#### keys()

#### m(method)

* **Return type:**
  `Callable`[[`ParamSpec`(`P`)], `TemporalVar`[`TypeVar`(`T`)]]

#### on_crossing(value, action=None, direction='both', terminal=False)

Execute the specified action when the signal crosses the specified value in the specified direction.
:type value: `Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]
:param value: Value to be crossed to trigger the event action.
:type action: `Union`[`Action`, `Callable`]
:param action: Action that is triggered when the crossing condition is met.
:type direction: `Literal`[`'rising'`, `'falling'`, `'both'`]
:param direction: Direction of the crossing that will trigger the event.
:type terminal: `Union`[`bool`, `int`]
:param terminal: If True, the simulation will terminate when the crossing occurs. If it is an integer, it
specifies the number of occurrences of the event after which the simulation will be terminated.
:rtype: `Event`
:return: Crossing event

#### *property* output_type

#### save(name)

Save the temporal variable with a name.

* **Parameters:**
  **name** (`str`) – Key to retrieve the variable.
* **Return type:**
  `None`

#### *property* shape

#### *property* t *: ndarray[tuple[int, ...], dtype[\_ScalarType_co]]*

#### to_plot(name=None)

Add the variable to the plotted data on solve.

* **Parameters:**
  **name** (`str`) – Name of the variable in the legend of the plot.
* **Return type:**
  `None`

#### *property* values *: ndarray[tuple[int, ...], dtype[\_ScalarType_co]]*
