"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

class ForwardEulerOutput(DenseOutput):
    """
    Dense output object for Forward Euler: performs linear interpolation 
    over the most recent step [t_old, t].
    Used by solve_ivp for querying y(tq) within a step via sol.sol(tq).
    """
    def __init__(self, t_old, t, y_old, y):
        super().__init__(t_old, t)
        self._y_old = np.asarray(y_old, dtype=float).reshape(-1)
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self._dt = float(t - t_old)

    def _call_impl(self, t):
        # Supports scalar or array queries; if array, return shape (n_states, m)
        if self._dt == 0.0:
            # Degenerate step: constant interpolation
            if np.ndim(t) == 0:
                return self._y.copy()                    
            else:
                m = np.asarray(t).size
                return self._y[:, None].repeat(m, axis=1)

        alpha = (np.asarray(t) - self.t_old) / self._dt    
        if np.ndim(alpha) == 0:
            return self._y_old + alpha * (self._y - self._y_old)

        # Vectorized query: construct (n, m)
        y0 = self._y_old[:, None]
        y1 = self._y[:, None]
        A  = alpha[None, :]
        return y0 + A * (y1 - y0)


class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Forward Euler ODE solver compatible with scipy.integrate.solve_ivp.

    Conventions and requirements:
    - Default step size h = (t_bound - t0) / 100 if not provided via solve_ivp(..., h=...)
    - direction = +1 
    - _dense_output_impl(self) returns a ForwardEulerOutput instance
    """

    # Explicitly specify direction as +1
    direction = +1

    def __init__(self, fun, t0, y0, t_bound, vectorized=False, h=None, **kwargs):
        # Initialize parent class
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, support_complex=False)

        # Step size setup: if not provided, divide the interval into 100 equal steps
        self.h = (t_bound - t0) / 100.0 if h is None else float(h)
        if self.h <= 0:
            raise ValueError("ForwardEuler requires a positive step size h.")

        # Forward Euler does not use Jacobian or LU decomposition
        self.njev = 0
        self.nlu = 0

        # Save previous step data for dense output
        self._t_old = self.t
        self._y_old = self.y.copy()

    def _step_impl(self):
        """
        Perform one Forward Euler step.
        Returns True if successfully advanced one step; 
        returns False when the integration reaches t_bound.
        """
        # Do not exceed t_bound
        h = min(self.h, self.t_bound - self.t)
        if h <= 0.0:
            return True, None 

        self._t_old = self.t
        self._y_old = self.y.copy()
    
        f = self.fun(self.t, self.y)
        self.nfev += 1

        self.y = self.y + h * f
        self.t = self.t + h

        return True, None 

    def _dense_output_impl(self):
        """
        Return the dense output for the most recent step [t_old, t] 
        (ForwardEulerOutput). 
        This enables stepwise interpolation via sol.sol(tq) in solve_ivp.
        """
        return ForwardEulerOutput(self._t_old, self.t, self._y_old, self.y)
