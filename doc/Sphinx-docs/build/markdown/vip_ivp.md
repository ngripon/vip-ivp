# vip_ivp package

### vip_ivp.api.clear()

Clear the current solver’s stored information.

* **Return type:**
  `None`

### vip_ivp.api.create_scenario(scenario_table, time_key, interpolation_kind='linear', sep=',')

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
  * **interpolation_kind** (*str* *,* *optional*) – The kind of interpolation to use. Default is “linear”. This determines how values are
    interpolated between time points.
  * **sep** (*str* *,* *optional*) – The separator to use when reading CSV files. Default is a comma.
* **Returns:**
  A dictionary of TemporalVar objects representing the scenario, where the keys are the variables and the
  values are the corresponding TemporalVar instances.
* **Return type:**
  Dict[Any, TemporalVar]
* **Raises:**
  **ValueError** – If the input file type is unsupported or the input type is invalid.

### vip_ivp.api.create_source(value)

Create a source signal from a temporal function or a scalar value.

* **Parameters:**
  **value** (`Union`[`Callable`[[`Union`[`float`, `ndarray`]], `TypeVar`(`T`)], `TypeVar`(`T`)]) – A function f(t) or a scalar value.
* **Return type:**
  `TemporalVar`[`TypeVar`(`T`)]
* **Returns:**
  The created TemporalVar.

### vip_ivp.api.delay(input_value, n_steps, initial_value=0)

* **Return type:**
  `TemporalVar`[`TypeVar`(`T`)]

### vip_ivp.api.differentiate(input_value, initial_value=0)

* **Return type:**
  `TemporalVar`[`float`]

### vip_ivp.api.explore(fun, t_end, bounds=(), time_step=None, title='')

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

### vip_ivp.api.export_file(filename, variables=None, file_format=None)

* **Return type:**
  `None`

### vip_ivp.api.export_to_df(variables=None)

* **Return type:**
  `DataFrame`

### vip_ivp.api.f(func)

* **Return type:**
  `Callable`[[`ParamSpec`(`P`)], `TemporalVar`[`TypeVar`(`T`)]]

### vip_ivp.api.get_events()

* **Return type:**
  `List`[`Event`]

### vip_ivp.api.get_var(var_name)

Retrieve a saved TemporalVar by its name.

* **Parameters:**
  **var_name** (`str`) – The name of the saved TemporalVar.
* **Return type:**
  `TemporalVar`
* **Returns:**
  The retrieved TemporalVar.

### vip_ivp.api.integrate(input_value, x0, minimum=None, maximum=None)

Integrate the input value starting from the initial condition x0.

* **Parameters:**
  * **minimum** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]) – Lower integration bound. Can be a TemporalVar
  * **maximum** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]) – Higher integration bound. Can be a TemporalVar
  * **input_value** (`Union`[`TemporalVar`[`TypeVar`(`T`)], `TypeVar`(`T`)]) – The value to be integrated, can be a TemporalVar or a number.
  * **x0** (`TypeVar`(`T`)) – The initial condition for the integration.
* **Return type:**
  `IntegratedVar`
* **Returns:**
  The integrated TemporalVar.

### vip_ivp.api.loop_node(shape=None)

Create a loop node. Loop node can accept new inputs through its “loop_into()” method after being instantiated.

* **Return type:**
  `LoopNode`
* **Returns:**
  The created LoopNode.

### vip_ivp.api.new_system()

Create a new solver system.

* **Return type:**
  `None`

### vip_ivp.api.plot()

Plot the variables that have been marked for plotting.

* **Return type:**
  `None`

### vip_ivp.api.save(\*args)

Save the given TemporalVars with their variable names.

* **Parameters:**
  **args** (`TemporalVar`) – TemporalVars to be saved.
* **Raises:**
  **ValueError** – If any of the arguments is not a TemporalVar.
* **Return type:**
  `None`

### vip_ivp.api.set_interval(action, delay)

* **Return type:**
  `Event`

### vip_ivp.api.set_timeout(action, delay)

* **Return type:**
  `Event`

### vip_ivp.api.solve(t_end, time_step=0.1, method='RK45', t_eval=None, include_events_times=True, \*\*options)

Solve the equations of the dynamical system through an integration scheme.

* **Parameters:**
  * **include_events_times** (`bool`) – If true, include time points at which events are triggered.
  * **t_end** (`float`) – Time at which the integration stops.
  * **method** – Integration method to use. Default is ‘RK45’.
  * **time_step** (`Optional`[`Number`]) – Time step for the integration. If None, use points selected by the solver.
  * **t_eval** (`Union`[`List`, `ndarray`]) – Times at which to store the computed solution. If None, use points selected by the solver.
  * **options** – Additional options for the solver.
* **Return type:**
  `None`

### vip_ivp.api.where(condition, a, b)
