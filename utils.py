class CasadiModel:
    states: ca.MX  # x
    inputs: ca.MX  # u
    sampling_time: Union[ca.MX, float, None] = field(default=None)
    outputs: Optional[ca.MX] = field(default=None)  # y
    state_derivative: Optional[ca.MX] = field(default=None)  # x_dot
    next_state: Optional[ca.MX] = field(default=None)  # x_new
    parameter: Optional[ca.MX] = field(default=None)  # p

    is_discrete: bool = field(init=False)
    has_outputs: bool = field(init=False)
    has_parameter: bool = field(init=False)

    _f_ca_function: ca.Function = field(init=False)  # [x,u (,p)]->[x_dot]
    _f_d_ca_function: Optional[ca.Function] = field(init=False)  # [x,u,(,p)]->[x_new]
    _f_d_approx_ca_function: Optional[ca.Function] = field(init=False)  # [x,u,(,p, h)]->[x_new]
    _g: Optional[ca.Function] = field(init=False)  # [x,u (,p)]->[y]

    nx: int = field(init=False)
    nu: int = field(init=False)
    np: int = field(init=False)
    ny: int = field(init=False)

    state_names: List[str] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    discretization_method: Optional[str] = 'rk4'
    discretization_steps: Optional[int] = 1

    def __post_init__(self):
        self.is_discrete = self._check_discrete()
        self._set_system_dimensions()
        self._create_casadi_functions()
        self._set_names()

    def _check_discrete(self) -> bool:
        if self.state_derivative is None and self.next_state is None:
            raise ValueError(f'Missing dynamical equation. For a continuous model specify state_derivative and'
                             f'for a discrete model specify next_state.')
        elif self.state_derivative is not None and self.next_state is None:
            is_discrete = False
        elif self.state_derivative is None and self.next_state is not None:
            is_discrete = True
        else:
            raise ValueError(f'Too many dynamical equations. For a continuous model specify state_derivative and'
                             f'for a discrete model specify next_state you cannot specify both.')
        return is_discrete

    def _set_system_dimensions(self):
        self.nx = self.states.shape[0]
        self.nu = self.inputs.shape[0]
        if self.outputs is not None:
            self.has_outputs = True
            self.ny = self.outputs.shape[0]
        else:
            self.has_outputs = False
            self.ny = 0

        if self.parameter is not None:
            self.has_parameter = True
            self.np = self.parameter.shape[0]
        else:
            self.has_parameter = False
            self.np = 0

    def _discretize_model_dynamics(self):
        if self.has_parameter is False:
            f = lambda x, u, p: self._f_ca_function(x, u)
        else:
            f = lambda x, u, p: self._f_ca_function(x, u, p)
        Ts = ca.MX.sym('Ts')

        x_new = self.states
        for i in range(self.discretization_steps):
            if self.discretization_method == 'rk4':
                x_new = rk4(f, Ts / self.discretization_steps, x_new, self.inputs, self.parameter)
            elif self.discretization_method == 'euler':
                x_new = euler(f, Ts / self.discretization_steps, x_new, self.inputs, self.parameter)
            else:
                raise NotImplementedError()
        if self.has_parameter:
            input_tuple = self.states, self.inputs, self.parameter
        else:
            input_tuple = self.states, self.inputs
        self._f_d_approx_ca_function = ca.Function('f_d_approx', [*input_tuple, Ts], [x_new])

    def set_sampling_time(self, sampling_time: Union[ca.MX, float, None]):
        if self.is_discrete:
            raise RuntimeError('sampling time of discrete model cannot be set.')
        self.sampling_time = sampling_time
        self._discretize_model_dynamics()

    def _create_casadi_functions(self):
        if self.has_parameter:
            input_tuple = self.states, self.inputs, self.parameter
        else:
            input_tuple = self.states, self.inputs

        if self.is_discrete:
            self._f_d_ca_function = ca.Function('f_d', [*input_tuple], [self.next_state])
            self._f_d_approx_ca_function = None
        else:
            self._f_d_ca_function = None
            self._f_ca_function = ca.Function('f', [*input_tuple], [self.state_derivative])
            self._discretize_model_dynamics()

        if self.has_outputs:
            self._g_ca_function = ca.Function('g', [*input_tuple], [self.outputs])
        else:
            self._g_ca_function = None

    def _set_names(self):
        names = [self.state_names, self.input_names, self.parameter_names, self.output_names]
        quantities = [self.states, self.inputs, self.parameter, self.outputs]
        abbreviations = ['x', 'u', 'p', 'y']
        for i in range(len(names)):
            if not names[i] and quantities[i] is not None:
                names[i] = [abbreviations[i] + f'_{j}' for j in range(quantities[i].shape[0])]
        self.state_names, self.input_names, self.parameter_names, self.output_names = tuple(names)

    def _check_function_call_inputs(self, x, u, p) -> Tuple[CasadiModelFunctionInputType, Tuple]:
        if self.has_parameter:
            input_tuple = x, u, p
            if p is None:
                raise RuntimeError('parameter not specified! you need to input a parameter vector.')
        else:
            input_tuple = x, u
            if p is not None:
                raise RuntimeError('you cannot specify a parameter for a model that has no parameter')

        if isinstance(x, ca.MX) or isinstance(u, ca.MX) or isinstance(p, ca.MX):
            input_type = CasadiModelFunctionInputType.SYMBOLIC
        else:
            if len(x.shape) == 1:
                assert x.shape == (self.nx,)
                assert u.shape == (self.nu,)
                if self.has_parameter and p is not None:
                    assert p.shape == (self.np,)
                input_type = CasadiModelFunctionInputType.NUMPY
            else:
                raise RuntimeError('invalid input type')
        return input_type, input_tuple
# Xiqiao Zhang
# 2024/7/26
