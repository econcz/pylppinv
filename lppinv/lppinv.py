import numbers
import numpy                           as np
from   typing          import Sequence
from   collections.abc import Sequence as sq
from   clsp            import CLSP

class LPPinvInputError(Exception):
    """
    Exception class for LPPinv-related input errors.

    Represents internal failures in Linear Programming via Regularized Least
    Squares routines due to malformed or incompatible input. Supports
    structured messaging and optional diagnostic augmentation.

    Parameters
    ----------
    message : str, optional
        Description of the error. Defaults to a generic LPPinv message.

    code : int or str, optional
        Optional error code or identifier for downstream handling.

    Usage
    -----
    raise LPPinvInputError("A_ub and b_ub are incompatible", code=201)
    """
    
    def __init__(self, message: str = "An error occurred in LPPinv",
                 code: int | str | None = None):
        self.message = message
        self.code    = code
        full_message = f"{message} (Code: {code})" if code is not None         \
                                                   else message
        super().__init__(full_message)
        
    def __str__(self) -> str:
        return self.message if self.code is None                               \
                            else f"{self.message} [Code: {self.code}]"
    
    def as_dict(self) -> dict:
        """
        Return the error as a dictionary for structured logging or JSON output.
        """
        return {"error": self.message, "code": self.code}

def lppinv(
    c:              Sequence[float]                         | None = None,
    A_ub:           Sequence[Sequence[float]]               | None = None,
    b_ub:           Sequence[float]                         | None = None,
    A_eq:           Sequence[Sequence[float]]               | None = None,
    b_eq:           Sequence[float]                         | None = None,
    non_negative:   bool                                           = True,
    bounds:         list[tuple[float | None, float | None]] |                  \
                         tuple[float | None, float | None]  | None = None,
    replace_value:  float                                   | None = np.nan,
    tolerance:      float = np.sqrt(np.finfo(float).eps),
    cond_tolerance: float                                   | None = None,
    slack_sign:     bool                                           = True,
    *args, **kwargs
) -> CLSP:
    """
    Solve a linear program via Convex Least Squares Programming (CLSP)
    estimator.

    The constraints are written in the CLSP canonical form
    [A_ub | I; A_eq | 0] @ [x; s] = [b_ub; b_eq], where s are the slacks of
    the inequality rows. The bounds on x and the sign of s (s ≥ 0) are
    enforced exactly in both steps of CLSP; the rows themselves are fitted
    in the least-squares sense (exactly if the constraints are feasible).
    The objective c is not optimized: for alpha = 1, the result is the
    minimum-norm point (x, s) among the bounded least-squares solutions.

    Parameters (SciPy linprog-compatible)
    -------------------------------------
    c : array_like of shape (p,), optional
        Objective function coefficients. Accepted for API parity; not used
        by CLSP.
    A_ub : array_like of shape (i, p), optional
        Matrix for inequality constraints A_ub @ x <= b_ub.
    b_ub : array_like of shape (i,), optional
        Right-hand side vector for inequality constraints. Rows with
        b_ub = +inf impose no restriction and are dropped; -inf (no
        feasible x) raises an LPPinvInputError.
    A_eq : array_like of shape (j, p), optional
        Matrix for equality constraints A_eq @ x = b_eq.
    b_eq : array_like of shape (j,), optional
        Right-hand side vector for equality constraints. ±inf raises an
        LPPinvInputError.
    non_negative : bool, default = True
        If False, no default nonnegativity is applied.
    bounds : sequence of (low, high), optional
        Bounds on variables, passed to CLSP as `bounds` and enforced
        exactly. If a single tuple (low, high) is given, it is applied to
        all variables. None denotes an unbounded side (a lower bound of 0
        if `non_negative` is True). If None, defaults to (0, None) for each
        variable (non-negativity).
    replace_value : float or None, default = np.nan
        Final replacement value for any cell in the returned CLSP.x that
        violates the specified bounds by more than the given tolerance
        (a safeguard; the bounds are enforced exactly by CLSP).
    tolerance : float, optional
        Tolerance of CLSP (bounded least squares, Step 2 verification) and
        of the bounds in `replace_value`. Default is the square root of
        machine epsilon.
    cond_tolerance : float or None, default = None
        Singular-value cutoff for the custom condition number function.
        If None, the implementation uses an internal relative cutoff of 1e-14.
    slack_sign : bool, default = True
        If True, the slacks of the inequality rows must be ≥ 0, so that
        A_ub @ x <= b_ub is free on its feasible side and a least-squares
        row on its infeasible side. If False, the slacks are free and the
        inequality rows impose no sign restriction.

    Parameters (CLSP, passed in **kwargs)
    -------------------------------------
    r : int, default = 1
        Number of refinement iterations (no effect without M rows).
    Z : np.ndarray or None
        Diagonal (0/1) projector on [x; s] for the Bott–Duffin inverse.
    rcond : float or bool, default = False
        Relative cutoff on the singular values of the canonical matrix.
    iteration_limit : int, default = 50
        Maximum number of passes of the bounded least-squares step.
    final : bool, default = True
        If True, the second step of CLSP is solved.
    alpha : float or None, default = None
        Regularization parameter in [0, 1] (0: Lasso, 1: Ridge, else
        Elastic Net), a single value; None means 1.
    scale : bool, default = False
        If True, the columns of the canonical matrix are equilibrated.
    solver, other keyword arguments
        CVXPY solver and options for the second step of CLSP.

    Returns
    -------
    CLSP
        The fitted CLSP object. Consult https://pypi.org/project/pyclsp/

    Raises
    ------
    LPPinvInputError
        If the constraint sets, bounds or alpha are malformed.
    """
    # assert conformability of constraint sets (A_ub, b_ub) and (A_eq, b_eq)
    if  not ((A_ub is not None and b_ub is not None)  or
             (A_eq is not None and b_eq is not None))                          \
        or  ((A_ub is None) ^ (b_ub is None))                                  \
        or  ((A_eq is None) ^ (b_eq is None)):
        raise LPPinvInputError("At least one complete constraint set "
                               "(A_ub, b_ub) or (A_eq, b_eq) must be "
                               "provided.")
    if  A_ub is not None:
        A_ub = np.asarray(A_ub, dtype=np.float64)
        if A_ub.ndim == 1:
            A_ub = A_ub.reshape(1, -1)
        b_ub = np.asarray(b_ub, dtype=np.float64).reshape(-1, 1)
        if np.any(b_ub == -np.inf):                    # x ≤ -inf: infeasible
            raise LPPinvInputError("b_ub must not contain -inf (no x "
                                   "satisfies A_ub @ x <= -inf).")
        if A_ub.shape[0] != b_ub.shape[0]:
            raise LPPinvInputError(f"A_ub and b_ub must have the same number "
                                   f"of rows: "
                                   f"{A_ub.shape[0]} vs {b_ub.shape[0]}")
        n_vars = A_ub.shape[1]                         # number of variables
    if  A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=np.float64)
        if A_eq.ndim == 1:
            A_eq = A_eq.reshape(1, -1)
        b_eq = np.asarray(b_eq, dtype=np.float64).reshape(-1, 1)
        if np.any(np.isinf(b_eq)):                     # A_eq @ x = ±inf
            raise LPPinvInputError("b_eq must not contain inf or -inf.")
        if A_eq.shape[0] != b_eq.shape[0]:
            raise LPPinvInputError(f"A_eq and b_eq must have the same number "
                                   f"of rows: "
                                   f"{A_eq.shape[0]} vs {b_eq.shape[0]}")
        n_vars = A_eq.shape[1]                         # number of variables

    # (bounds), (b) Normalize the bounds, construct the right-hand side
    if  bounds is None:                                # normalize bounds
        bounds = (0 if non_negative else None, None)
    if  isinstance(bounds, tuple) and len(bounds) == 2                         \
        and all(v is None or isinstance(v, numbers.Real) for v in bounds):
        bounds = [bounds] * n_vars                     # replicate (low, high)
    elif   isinstance(bounds, sq):
        bounds = list(bounds)
        if len(bounds) > 1 and len(bounds) != n_vars:
            raise LPPinvInputError(f"Bounds length {len(bounds)} does not "
                                   f"match number of variables {n_vars}.")
        elif len(bounds) == 1:
            bounds = bounds * n_vars                   # replicate (low, high)
    if non_negative and any((l is not None and l < 0) or
                            (h is not None and h < 0) for l, h in bounds):
        raise LPPinvInputError("Negative lower or upper bounds are not "
                               "allowed in linear programs.")
    bounds = [(l if l is not None else (0 if non_negative else None), h)
              for l, h in bounds]                      # x ≥ 0 by default
    b = np.empty((0, 1))
    if b_ub is not None:
        b = b_ub
    if b_eq is not None:
        b = np.vstack([b, b_eq])

    # (C), (S) Construct conformable blocks for the design matrix A
    if  A_ub is not None and A_eq is not None:
        if A_ub.shape[1] != A_eq.shape[1]:
            raise LPPinvInputError(f"A_ub and A_eq must have the same number "
                                   f"of columns: "
                                   f"{A_ub.shape[1]} vs {A_eq.shape[1]}")
    C = np.empty((0, n_vars))
    S = np.empty((0, 0))
    if  A_ub is not None:
        C = A_ub
        S = np.eye(A_ub.shape[0])
    if  A_eq is not None:
        C = np.vstack([C, A_eq])
        S = np.vstack([S, np.zeros((A_eq.shape[0], S.shape[1]))])

    # (result) Perform the estimation
    assert C.shape[0] == S.shape[0] == b.shape[0],                             \
           f"Row mismatch: C={C.shape}, S={S.shape}, b={b.shape}"
    if  isinstance(kwargs.get("alpha"), (list, tuple, np.ndarray)):
        raise LPPinvInputError("alpha must be a single value in [0, 1]; "
                               "selection by NRMSE was removed in 5.0.0.")
    kw = {k: v for k, v in kwargs.items() if k not in {"problem", "C", "S",
                                                       "M", "b"}}
    finite_rows  = b[:, 0] != np.inf                   # drop b_ub = +inf rows
    nonzero_cols = ~np.all(S[finite_rows, :] == 0,
                           axis=0)                     # reduce S width
    if  not np.any(finite_rows):
        raise LPPinvInputError("At least one constraint with a finite "
                               "right-hand side must be provided.")
    result   = CLSP().solve(*args,
                            problem='general', C=(C[finite_rows, :]),
                                               S=(S[finite_rows, :]
                                                  )[:,nonzero_cols],
                                               b=(b[finite_rows, :]),
                                          bounds=bounds,
                                      slack_sign=slack_sign,
                                       tolerance=tolerance,
                                  cond_tolerance=cond_tolerance, **kw)

    # (result) Replace out-of-bound values with `replace_value`
    x        = result.x.reshape(-1, 1)
    x_lb     = result.x_lb.reshape(-1, 1)
    x_ub     = result.x_ub.reshape(-1, 1)
    x[(x < x_lb - tolerance) | (x > x_ub + tolerance)] = replace_value
    result.x = x.reshape(result.x.shape)

    return result
