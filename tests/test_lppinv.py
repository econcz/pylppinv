"""
Regression and acceptance tests for pylppinv 5.0.0.

The references are computed with numpy, scipy and cvxpy only. Runs under
pytest or as a plain script: `python tests/test_lppinv.py`. With the
environment variable LPPINV_TEST_INSTALLED=1, the installed package is tested
instead of the repo copy; CLSP_SRC=<path> prepends a source tree of pyclsp
to sys.path.
"""
import os
import sys
import traceback
import warnings
from   pathlib        import Path
import numpy          as np
import cvxpy          as cp
from   scipy.optimize import lsq_linear

ROOT      = Path(__file__).resolve().parents[1]
INSTALLED = os.environ.get("LPPINV_TEST_INSTALLED") == "1"
if os.environ.get("CLSP_SRC"):                       # pyclsp source tree
    sys.path.insert(0, os.environ["CLSP_SRC"])
if not INSTALLED:                                    # test the repo copy
    sys.path.insert(0, str(ROOT))
import clsp                                          # noqa: E402
import lppinv         as pkg                         # noqa: E402
from   lppinv         import lppinv                  # noqa: E402
from   lppinv.lppinv  import LPPinvInputError        # noqa: E402
from   clsp.errors    import CLSPError               # noqa: E402

TOL      = 1e-8                                      # numerical tolerance
A_UB     = np.array([[1.0, 1.0], [1.0, -1.0]])       # small LP
B_UB     = np.array([4.0, 1.0])
A_EQ     = np.array([[1.0, 2.0]])
B_EQ     = np.array([5.0])
CLARABEL = dict(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                tol_feas=1e-12, tol_ktratio=1e-10, max_iter=500)

# Ancillary functions (independent references)
def _canonical(A_ub, b_ub, A_eq, b_eq):
    """[A_ub | I; A_eq | 0] and [b_ub; b_eq]."""
    n_ub = A_ub.shape[0] if A_ub is not None else 0
    n    = (A_ub if A_ub is not None else A_eq).shape[1]
    A    = np.zeros((0, n + n_ub))
    if A_ub is not None:
        A = np.vstack([A, np.hstack([A_ub, np.eye(n_ub)])])
    if A_eq is not None:
        A = np.vstack([A, np.hstack([A_eq, np.zeros((A_eq.shape[0], n_ub))])])
    b = np.concatenate([np.ravel(v) for v in (b_ub, b_eq) if v is not None])
    return A, b

def _z_bounds(lb, ub, n_ub):
    """Bounds on z = [x; s] with s ≥ 0."""
    return (np.r_[lb, np.zeros(n_ub)], np.r_[ub, np.full(n_ub, np.inf)])

def _bvls(A, b, lb, ub):
    """Bounded least squares (BVLS): z and the residual norm ‖b - Az‖."""
    res = lsq_linear(A, b, bounds=(lb, ub), method="bvls",
                     tol=1e-14, max_iter=10_000)
    return res.x, np.linalg.norm(b - A @ res.x)

def _min_norm_feasible(A, w, lb, ub):
    """argmin ‖z‖² s.t. Az = w, lb ≤ z ≤ ub (CLARABEL, tight)."""
    z   = cp.Variable(A.shape[1])
    con = [A @ z == w]
    con += [z[i] >= lb[i] for i in np.flatnonzero(np.isfinite(lb))]
    con += [z[i] <= ub[i] for i in np.flatnonzero(np.isfinite(ub))]
    cp.Problem(cp.Minimize(cp.sum_squares(z)), con).solve(**CLARABEL)
    return z.value

def _random_lp(seed, n=8, n_ub=4, n_eq=3, feasible=True):
    """Random LP-type constraints, feasible (x0 ≥ 0) or not for x ≥ 0."""
    rng  = np.random.default_rng(seed)
    A_ub = rng.standard_normal((n_ub, n))
    A_eq = rng.standard_normal((n_eq, n))
    x0   = rng.uniform(0.0, 2.0, n)
    b_ub = A_ub @ x0 + rng.uniform(0.0, 1.0, n_ub)
    b_eq = A_eq @ x0
    if not feasible:                                 # A_ub·x ≥ 0 > b_ub
        A_ub = np.abs(A_ub)
        b_ub = -rng.uniform(0.5, 1.5, n_ub)
    return A_ub, b_ub, A_eq, b_eq

def _lp(**kw):
    """Small LP, x ≥ 0."""
    kw = dict(A_ub=A_UB, b_ub=B_UB, A_eq=A_EQ, b_eq=B_EQ) | kw
    return lppinv(**kw)

def _raises(error, func, text=""):
    try:
        func()
    except error as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f"no {error.__name__}")

def _close(a, b, tol=TOL):
    return np.allclose(np.ravel(a), np.ravel(b), rtol=0.0, atol=tol)

# 0. Package under test
def test_imports_package_under_test():
    path = Path(pkg.__file__).resolve().parent
    assert (path == ROOT / "lppinv") != INSTALLED
    assert pkg.__version__ == "5.0.0"
    assert int(clsp.__version__.split(".")[0]) >= 5

# 1. Canonical form: C = [A_ub; A_eq], S = [I; 0], bounds, slack_sign
def test_canonical_form():
    c    = _lp()
    A, b = _canonical(A_UB, B_UB, A_EQ, B_EQ)
    assert _close(c.A, A, 0.0) and _close(c.b, b, 0.0)
    assert tuple(c.C_idx) == (3, 2) and tuple(c.S_idx) == (2, 4)
    assert _close(c.x_lb, [0.0, 0.0], 0.0) and np.all(np.isinf(c.x_ub))
    assert c.slack_sign is True and c.x.shape == (2, 1) and c.y.shape == (2, 1)

def test_canonical_form_equality_only():
    c = lppinv(A_eq=A_EQ, b_eq=B_EQ)
    assert c.A.shape == (1, 2) and c.y.shape == (0, 1)
    assert tuple(c.S_idx) == (2, 2)

# 2. Feasible LP-type problem with x ≥ 0 and ≤ rows
def test_feasible_lp_min_norm_point():
    for seed in range(5):
        A_ub, b_ub, A_eq, b_eq = _random_lp(seed)
        c      = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        x, s   = c.x.ravel(), c.y.ravel()
        assert c.status["step1"] == "converged", c.status
        assert c.status["final"] in {"optimal", "optimal_inaccurate"}
        assert c.rho < TOL and c.residual < 1e-6
        assert np.all(A_ub @ x <= b_ub + 1e-6) and _close(A_eq @ x, b_eq, 1e-6)
        assert np.all(x >= -TOL) and np.all(s >= -TOL)
        A, b   = _canonical(A_ub, b_ub, A_eq, b_eq)
        lb, ub = _z_bounds(np.zeros(8), np.full(8, np.inf), 4)
        assert _close(c.z, _min_norm_feasible(A, b, lb, ub), 1e-5)

def test_objective_not_used():
    c0 = _lp()
    for obj in ([1.0, 1.0], [-1.0, -2.0], [5.0, -3.0]):
        assert _close(_lp(c=obj).z, c0.z, 0.0)

# 3. Infeasible or inconsistent constraints: soft rows
def test_infeasible_soft_rows():
    c   = lppinv(A_ub=[[1.0, 1.0]], b_ub=[1.0],      # x1 + x2 ≤ 1
                 A_eq=[[1.0, 1.0]], b_eq=[3.0])      # x1 + x2 = 3
    assert c.status["step1"] == "converged"
    assert abs(c.rho - np.sqrt(2.0)) < TOL           # rows traded off
    assert abs(c.residual - c.rho) < 1e-6
    assert abs(c.x.sum() - 2.0) < 1e-6 and abs(c.y[0, 0]) < 1e-6
    assert np.all(c.x >= -TOL)

def test_infeasible_random_rho_is_bvls_residual():
    for seed in range(5):
        A_ub, b_ub, A_eq, b_eq = _random_lp(seed, feasible=False)
        c      = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        A, b   = _canonical(A_ub, b_ub, A_eq, b_eq)
        lb, ub = _z_bounds(np.zeros(8), np.full(8, np.inf), 4)
        _, rho = _bvls(A, b, lb, ub)
        assert rho > 1e-3
        assert abs(c.rho - rho) < 1e-8 * max(1.0, rho)
        assert abs(c.residual - c.rho) < 1e-6 * max(1.0, rho)
        assert c.status["bound_error"] <= TOL
        assert c.status["eq_error"] <= c.status["tol_v"]

# 4. Equality-only problem
def test_equality_only_unbounded_min_norm():
    rng = np.random.default_rng(4)
    A   = rng.standard_normal((3, 7))
    b   = rng.standard_normal(3)
    c   = lppinv(A_eq=A, b_eq=b, non_negative=False)
    assert _close(c.x, np.linalg.pinv(A) @ b)
    assert c.rho < TOL and c.status["rho"] == "pinv"

def test_equality_only_nonnegative():
    rng  = np.random.default_rng(5)
    A    = rng.standard_normal((3, 7))
    b    = A @ rng.uniform(0.0, 1.0, 7)
    c    = lppinv(A_eq=A, b_eq=b)
    assert np.all(c.x >= -TOL) and c.rho < TOL
    assert _close(c.x, _min_norm_feasible(A, b, np.zeros(7),
                                          np.full(7, np.inf)), 1e-5)

# 5. Bounds and slack signs
def test_bounds_and_slack_signs_satisfied():
    for seed in range(10):
        rng  = np.random.default_rng(100 + seed)
        A_ub, b_ub, A_eq, b_eq = _random_lp(seed, feasible=seed % 2 == 0)
        lo   = -rng.uniform(0.0, 1.0, 8)
        hi   = rng.uniform(0.5, 2.0, 8)
        c    = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      non_negative=False, bounds=list(zip(lo, hi)))
        x, s = c.x.ravel(), c.y.ravel()
        assert np.all(x >= lo - TOL) and np.all(x <= hi + TOL)
        assert np.all(s >= -TOL) and not np.any(np.isnan(x))
        assert c.status["step1"] in {"converged", "bvls fallback"}

def test_bounds_forms():
    ref = _lp(bounds=[(0, 3), (0, 3)]).z
    for bounds in [(0, 3), [(0, 3)], ((0, 3), (0, 3)), [(None, 3)] * 2]:
        assert _close(_lp(bounds=bounds).z, ref, 0.0), bounds
    c   = _lp(bounds=[(None, 3), (1, None)])         # None lower: 0
    assert _close(c.x_lb, [0.0, 1.0], 0.0) and _close(c.x_ub, [3.0, np.inf])
    c   = _lp(bounds=(None, None), non_negative=False)
    assert np.all(np.isinf(c.x_lb)) and np.all(np.isinf(c.x_ub))
    _raises(LPPinvInputError, lambda: _lp(bounds=[(0, 1)] * 3), "Bounds")
    _raises(LPPinvInputError, lambda: _lp(bounds=(-1, 1)), "Negative")
    _raises(CLSPError, lambda: _lp(bounds=(2, 1)), "lo ≤ hi")

def test_fixed_variable_and_replace_value():
    c = _lp(bounds=[(1, 1), (0, None)])              # x1 = 1
    assert c.x[0, 0] == 1.0 and bool(c.fixed[0])
    assert not np.any(np.isnan(c.x))                 # nothing replaced

def test_slack_sign_false_free_slacks():
    c = lppinv(A_ub=[[1.0, 1.0]], b_ub=[1.0], A_eq=[[1.0, 1.0]], b_eq=[3.0],
               slack_sign=False)
    assert c.slack_sign is False
    assert c.y[0, 0] < -1.0                          # ≤ row absorbed
    assert c.rho < TOL and abs(c.x.sum() - 3.0) < 1e-6

# 6. alpha: a single value in [0, 1]
def test_alpha_validation():
    for alpha in ([0.0, 1.0], (0.5,), np.array([1.0])):
        _raises(LPPinvInputError, lambda: _lp(alpha=alpha), "single value")
    for alpha in (2.0, -0.1, np.nan, "1"):
        _raises(CLSPError, lambda: _lp(alpha=alpha), "single value")
    A_ub, b_ub, A_eq, b_eq = _random_lp(7, feasible=False)
    fits = [lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, alpha=alpha)
            for alpha in (0.0, 0.5, 1.0, None)]
    assert all(abs(c.residual - fits[2].residual) < 1e-5 for c in fits)
    assert fits[3].alpha == 1.0 and _close(fits[3].z, fits[2].z, 0.0)

# 7. Step 1 against scipy BVLS on the same canonical system
def test_step1_matches_scipy_bvls():
    for seed in range(8):
        rng    = np.random.default_rng(200 + seed)
        A_ub, b_ub, A_eq, b_eq = _random_lp(seed, n=6, n_ub=5, n_eq=4,
                                            feasible=False)
        hi     = rng.uniform(0.5, 3.0, 6)
        c      = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=[(0, h) for h in hi], final=False)
        A, b   = _canonical(A_ub, b_ub, A_eq, b_eq)
        lb, ub = _z_bounds(np.zeros(6), hi, 5)
        z, rho = _bvls(A, b, lb, ub)
        assert c.status["step1"] == "converged", c.status
        assert abs(np.linalg.norm(b - A @ c.zhat.ravel()) - rho) < 1e-8
        assert _close(A @ c.zhat.ravel(),            # unique fit
                      A @ z, 1e-6)
        assert np.all(c.zhat.ravel() >= lb - TOL)
        assert np.all(c.zhat.ravel() <= ub + TOL)

# 8. Options passed through to CLSP
def test_passthrough_options():
    c = _lp(final=False)
    assert c.final is False and c.status["final"] == "skipped"
    assert _close(c.z, c.zhat, 0.0)
    c = _lp(scale=True, rcond=1e-10, iteration_limit=7, tolerance=1e-10,
            cond_tolerance=1e-12, r=2)
    assert c.scale is True and c.status["scale"] is True and c.rcond == 1e-10
    assert c.iteration_limit == 7 and c.tolerance == 1e-10 and c.r_max == 2
    c = _lp(solver="CLARABEL", bounds=(0, 1))
    assert c.status["solver"] == "CLARABEL"
    c = _lp(alpha=0.0, bounds=(0, 1))
    assert c.status["solver"] == "ECOS" and c.alpha == 0.0
    c = _lp(C=np.eye(2), S=None, M=np.eye(2), b=np.ones(2), problem="ap")
    assert _close(c.z, _lp().z, 0.0)                 # ignored

# 9. Rows with an infinite right-hand side and input errors
def test_infinite_rows_dropped():
    c  = _lp(b_ub=[np.inf, 1.0])
    c1 = lppinv(A_ub=A_UB[1:], b_ub=B_UB[1:], A_eq=A_EQ, b_eq=B_EQ)
    assert c.A.shape == (2, 3) and _close(c.z, c1.z, 0.0)
    _raises(LPPinvInputError, lambda: lppinv(A_ub=A_UB, b_ub=[np.inf] * 2),
            "finite")
    _raises(CLSPError, lambda: _lp(b_ub=[np.nan, 1.0]), "finite")

def test_infinite_rows_rejected():
    # x ≤ -inf and A_eq @ x = ±inf are not vacuous: an input error
    _raises(LPPinvInputError, lambda: lppinv(A_ub=[[1.0]], b_ub=[-np.inf],
                                             A_eq=[[1.0]], b_eq=[1.0]),
            "-inf")
    for v in (np.inf, -np.inf):
        _raises(LPPinvInputError, lambda: lppinv(A_ub=[[1.0]], b_ub=[0.5],
                                                 A_eq=[[1.0]], b_eq=[v]),
                "inf")
    _raises(CLSPError, lambda: lppinv(A_ub=[[1.0]], b_ub=[0.5],
                                      A_eq=[[1.0]], b_eq=[np.nan]), "finite")
    c = lppinv(A_ub=[[1.0]], b_ub=[np.inf], A_eq=[[1.0]], b_eq=[1.0])
    assert _close(c.x, [1.0], 1e-10) and c.A.shape == (1, 1)

def test_input_errors():
    _raises(LPPinvInputError, lambda: lppinv(c=[1.0, 1.0]), "complete")
    _raises(LPPinvInputError, lambda: lppinv(A_ub=A_UB), "complete")
    _raises(LPPinvInputError, lambda: lppinv(A_eq=A_EQ, b_eq=[1.0, 2.0]),
            "same number of rows")
    _raises(LPPinvInputError, lambda: _lp(A_eq=np.ones((1, 3))),
            "same number of columns")

def test_no_warnings_on_regular_problems():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in range(3):
            A_ub, b_ub, A_eq, b_eq = _random_lp(seed, feasible=seed != 1)
            lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

# Plain-script runner
if __name__ == "__main__":
    tests  = [(k, f) for k, f in list(globals().items())
              if k.startswith("test_") and callable(f)]
    failed = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=-2)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
