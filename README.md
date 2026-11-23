# Linear Programming via Regularized Least Squares

The **Linear Programming via Regularized Least Squares (LPPinv)** is a two-stage estimation method that reformulates linear programs as structured least-squares problems. Based on the [Convex Least Squares Programming (CLSP)](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") framework, LPPinv solves linear inequality, equality, and bound constraints by (1) constructing a canonical constraint system and computing a pseudoinverse projection, enforcing the bounds on the variables and the signs of the inequality slacks by bounded-variable least squares, followed by (2) a convex-programming correction stage that selects, among the equally well-fitting solutions, the one closest to the pseudoinverse estimate under additional regularization (e.g., Lasso, Ridge, or Elastic Net).
LPPinv is intended for **underdetermined**, **ill-posed** and **infeasible** linear problems, for which standard solvers fail or return no solution: if the constraints are feasible, LPPinv returns a feasible point (for the default α = 1, the one of minimum norm); if they are not, it returns the least-squares compromise between the constraint rows. Please note that the objective `c` is accepted for compatibility with `scipy.optimize.linprog` but is **not optimized** (see [Limitations](#limitations)).
All calculations are performed in numpy.float64 precision. Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [Installation](#installation)
- [How LPPinv Works](#how-lppinv-works): [The Canonical Form](#the-canonical-form), [Step 1](#step-1-bounded-least-squares), [Step 2](#step-2-convex-correction)
- [Quick Example](#quick-example)
- [Examples](#examples):
  [1. LPPinv vs. an LP Solver](#1-lppinv-vs-an-lp-solver),
  [2. Infeasible Constraints](#2-infeasible-constraints-soft-rows),
  [3. Bounds and Slack Signs](#3-bounds-and-slack-signs),
  [4. Step 2 and α](#4-step-2-and-α),
  [5. Reading the Results](#5-reading-the-results)
- [User Reference](#user-reference)
- [Warnings and Troubleshooting](#warnings-and-troubleshooting)
- [Limitations](#limitations)
- [What's New in 5.0.0](#whats-new-in-500)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements) · [Bibliography](#bibliography) · [License](#license)

## Installation

```bash
pip install pylppinv
```

The package requires Python ≥ 3.11 (tested on Python 3.11, 3.12 and 3.13) and depends on numpy and [pyclsp](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") ≥ 5.0.0 (installed automatically, together with scipy, cvxpy and its solvers).

## How LPPinv Works

### The Canonical Form

`lppinv()` accepts the constraints of `scipy.optimize.linprog`: `A_ub` @ x ≤ `b_ub`, `A_eq` @ x = `b_eq` and `bounds` on x. Each inequality row receives a slack variable s ≥ 0, and the constraints are written as one linear system in the canonical form of CLSP:

```text
A z = b,      A = [ A_ub  I ]      z = [ x ]      b = [ b_ub ]
                  [ A_eq  0 ]          [ s ]          [ b_eq ]

lb ≤ x ≤ ub   (bounds, x ≥ 0 by default),      s ≥ 0   (slack_sign=True)
```

In the notation of pyclsp, LPPinv solves a general problem (`problem=""`) with `C` = [`A_ub`; `A_eq`], `S` = [I; 0] (S_ii = +1 encodes "≤"), no model rows (`M`), `bounds` on x and `slack_sign=True`. Rows with `b_ub` = +inf impose no restriction and are dropped together with their slacks; `b_ub` = −inf (no x satisfies the row) and an infinite `b_eq` raise an *LPPinvInputError*. x is returned as `model.x` and s as `model.y`. Since version 5.0.0, the bounds and the signs of the slacks are passed to pyclsp and enforced exactly in both steps; earlier versions appended the bounds as additional constraint rows and replaced out-of-bound values by `replace_value` afterwards.

Please note that the rows of `A` are fitted *in least squares*, as in pyclsp: if the constraints are feasible, they are met (up to the numerical tolerance); if they are infeasible, the misfit of the rows is traded off in least squares. An inequality row is free on its feasible side (its slack s ≥ 0 absorbs the difference) and a least-squares row on its infeasible side (s = 0); an equality row is a least-squares row. Only the bounds on x and the signs of the slacks are hard restrictions.

### Step 1: Bounded Least Squares

Step 1 of CLSP computes the unbounded estimate zhat0 = `A`_k⁺`b` (`model.zhat0`) from one SVD of `A` (`A`_k is `A` with the singular values below the cutoff `rcond` set to zero) and solves the bounded least-squares problem

```text
zhat ∈ argmin ‖b − A_k z‖₂   subject to   lb ≤ x ≤ ub,   s ≥ 0
```

by bounded-variable least squares (Lawson–Hanson, Stark–Parker), verified by the full KKT conditions, with `scipy.optimize.lsq_linear(..., method="bvls")` as a fallback (`model.zhat`, `model.status["step1"]`). The smallest residual norm ρ = ‖`b` − `A`_k·zhat‖₂ (`model.rho`) is 0 if the constraints are feasible (within the numerical range of `A`) and positive otherwise, i.e., ρ measures the infeasibility of the constraint system. The variables at a bound are reported in `model.fixed` (over z, i.e., x followed by s). See the [pyclsp README](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") for the details of the algorithm.

### Step 2: Convex Correction

If `final=True` (the default), Step 2 of CLSP keeps the fitted vector w* = `A`_k·zhat fixed and selects

```text
z* = argmin (1 − α)‖z − zhat0‖₁ + α‖z − zhat0‖₂²   subject to   A_k z = w*,   lb ≤ x ≤ ub,   s ≥ 0
```

All α attain the same residual ρ; α only selects among the equally well-fitting solutions. For α = 1 (Ridge, the default), z* is the minimum-norm point (x, s), i.e., the point with the smallest ‖x‖² + ‖s‖² among all bounded least-squares solutions: for feasible constraints, the feasible point of minimum norm. For α = 0 (Lasso) and 0 < α < 1 (Elastic Net), z* minimizes the L1 or mixed distance to zhat0. The solution is verified (`model.status["final"]`); if the solver fails, z = zhat is returned with a *RuntimeWarning*.

## Quick Example

Each example below is self-contained; the output shown was produced by running it with pylppinv 5.0.0 and pyclsp 5.0.0 (numpy 64-bit floats; the last digits of solver-dependent values may differ across platforms and solver versions).

```python
import numpy  as     np
from   lppinv import lppinv

# LPRLS/QPRLS, based on an underdetermined problem with x ≥ 0

seed   = 123456789
rng    = np.random.default_rng(seed)

# sample (dataset)
A_ub   = rng.normal(size=(50, 500))                          # underdetermined LP/QP matrix
A_eq   = rng.normal(size=(25, 500))                          # underdetermined LP/QP matrix
b_ub   = rng.normal(size=(50,   1))                          # may be inconsistent with A_ub
b_eq   = rng.normal(size=(25,   1))                          # may be inconsistent with A_eq
model  = lppinv(
             A_ub=A_ub, A_eq=A_eq, b_ub=b_ub, b_eq=b_eq,
             non_negative=True,                              # x ≥ 0 (default)
             r=1,                                            # a solution without refinement
             alpha=1.0                                       # unique, minimum-norm (α = 1)
         )

# results
x      = model.x.ravel()
s      = model.y.ravel()                                     # slacks of the A_ub rows
print("status:", model.status["step1"], "/", model.status["final"])
print("x hat: min =", round(x.min(), 4), " max =", round(x.max(), 4),
      " zeros =", int(np.sum(x == 0)), "of", x.size)
print("slacks: min =", round(s.min(), 4))
print("rho =", f"{model.rho:.1e}", " residual =", f"{model.residual:.1e}")
print("max(A_ub @ x - b_ub) =", f"{np.max(A_ub @ x - b_ub.ravel()):.1e}")
print("max|A_eq @ x - b_eq| =", f"{np.max(np.abs(A_eq @ x - b_eq.ravel())):.1e}")

model.summary(display=True)
```

Output:

```text
status: converged / optimal
x hat: min = 0.0  max = 0.1252  zeros = 263 of 500
slacks: min = 0.0
rho = 1.9e-14  residual = 2.4e-14
max(A_ub @ x - b_ub) = 3.8e-15
max|A_eq @ x - b_eq| = 6.2e-15
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                 [0, inf]
  Slack sign (S ≥ 0):             True
  Column scaling:                False
  Fixed variables:                 332

Numerical Stability:
  kappaC:                       2.2168
  kappaB:                       1.0000
  kappaA:                       2.2168
  rank:                             75

Goodness of Fit:
  r2_partial:                      nan
  nrmse:                     2.710e-15
  nrmse_partial:                   nan
  z_lower:        min= 0.000e+00  max=    0.1252  mean=    0.0161  sd=    0.0251
  z_upper:        min= 0.000e+00  max=    0.1252  mean=    0.0161  sd=    0.0251
  x_lower:        min= 0.000e+00  max=    0.1252  mean=    0.0175  sd=    0.0259
  x_upper:        min= 0.000e+00  max=    0.1252  mean=    0.0175  sd=    0.0259
  y_lower:        min= 0.000e+00  max=    0.0186  mean=    0.0022  sd=    0.0040
  y_upper:        min= 0.000e+00  max=    0.0186  mean=    0.0022  sd=    0.0040
```

The constraints are feasible (ρ ≈ 0): the estimate satisfies `A_ub` @ x ≤ `b_ub` and `A_eq` @ x = `b_eq` up to rounding, with x ≥ 0 and s ≥ 0 enforced exactly. Among the feasible points, Step 2 (α = 1) returns the one with the smallest ‖x‖² + ‖s‖². *Fixed variables* counts the components of z = (x, s) at a bound at the end of Step 1, which need not coincide with the zeros of the final x. The goodness-of-fit statistics of the summary refer to the constraint rows (there are no model rows, so `r2_partial` and `nrmse_partial` are *nan*).

## Examples

### 1. LPPinv vs. an LP Solver

```python
import numpy          as     np
from   scipy.optimize import linprog
from   lppinv         import lppinv

# LP: maximize x1 + 2 x2 subject to x1 + x2 ≤ 4, x1 - x2 ≤ 1, x1 + 3 x2 ≤ 9, x ≥ 0

c      = np.array([-1.0, -2.0])                              # linprog minimizes c @ x
A_ub   = np.array([[1.0,  1.0],
                   [1.0, -1.0],
                   [1.0,  3.0]])
b_ub   = np.array([4.0, 1.0, 9.0])
lp     = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None)) # the LP optimum (a vertex)
model  = lppinv(c=c, A_ub=A_ub, b_ub=b_ub)                   # c is accepted, but not used

# results
print("linprog: x =", np.round(lp.x, 4), " c @ x =", round(c @ lp.x, 4))
print("lppinv:  x =", np.round(model.x.ravel(), 4),
      " c @ x =", round(float(c @ model.x.ravel()), 4))
print("slacks s = b_ub - A_ub @ x:", np.round(model.y.ravel(), 4))
print("rho =", f"{model.rho:.1e}", " status:", model.status["final"])
```

Output:

```text
linprog: x = [1.5 2.5]  c @ x = -6.5
lppinv:  x = [2. 2.]  c @ x = -6.0
slacks s = b_ub - A_ub @ x: [0. 1. 1.]
rho = 1.9e-15  status: optimal
```

`linprog` returns the optimal vertex of the LP. LPPinv ignores `c` and returns the feasible point with the smallest ‖x‖² + ‖s‖², which lies on the first constraint (s₁ = 0) but is not a vertex of the feasible region. It is a feasible, but not an optimal, solution of the LP. LPPinv is therefore no substitute for an LP solver when the objective matters; it is useful when a regularized point of an underdetermined, ill-posed or infeasible linear system is sought.

### 2. Infeasible Constraints: Soft Rows

```python
import numpy  as     np
from   lppinv import lppinv

# Infeasible constraints: x1 + x2 ≤ 1 and x1 + x2 = 3, x ≥ 0

A_ub   = np.array([[1.0, 1.0]])
b_ub   = np.array([1.0])
A_eq   = np.array([[1.0, 1.0]])
b_eq   = np.array([3.0])
for weight in (1.0, 1.0e3):                                  # weight of the equality row
    model = lppinv(A_ub=A_ub, b_ub=b_ub,
                   A_eq=weight * A_eq, b_eq=weight * b_eq)
    x     = model.x.ravel()
    print(f"weight = {weight:g}: x = {np.round(x, 4)}, x1 + x2 = {x.sum():.4f}, "
          f"s = {model.y[0, 0]:.4f}")
    print(f"  rho = {model.rho:.4f}, residual = {model.residual:.4f}, "
          f"step1 = {model.status['step1']}, final = {model.status['final']}, "
          f"kappaA = {model.kappaA:.1f}")
```

Output:

```text
weight = 1: x = [1. 1.], x1 + x2 = 2.0000, s = 0.0000
  rho = 1.4142, residual = 1.4142, step1 = converged, final = optimal, kappaA = 3.2
weight = 1000: x = [1.5 1.5], x1 + x2 = 3.0000, s = 0.0000
  rho = 2.0000, residual = 2.0000, step1 = converged, final = optimal, kappaA = 1414.2
```

No x satisfies both rows. With equal weights, both rows are least-squares rows on their violated sides (s = 0), and the estimate is the compromise x1 + x2 = 2 with ρ = √2. Neither row is a hard constraint: scaling a row (and its right-hand side) by a large weight makes it nearly binding (here, the equality), at the cost of a larger κ(`A`); this is a penalty approach, and ρ then refers to the weighted system.

### 3. Bounds and Slack Signs

```python
import numpy  as     np
from   lppinv import lppinv

# Bounds and slack signs: x1 + x2 + x3 ≤ 2, x1 - x3 = 0.5, x2 = 2.5 (infeasible for x ≥ 0)

A_ub   = np.array([[1.0, 1.0,  1.0]])
b_ub   = np.array([2.0])
A_eq   = np.array([[1.0, 0.0, -1.0],
                   [0.0, 1.0,  0.0]])
b_eq   = np.array([0.5, 2.5])
for label, kw in [("x ≥ 0 (default)       ", dict()),
                  ("x2 ≤ 2, x3 = 0.25     ", dict(bounds=[(0, None), (0, 2),
                                                          (0.25, 0.25)])),
                  ("x ≥ 0, free slack     ", dict(slack_sign=False)),
                  ("x unbounded, s ≥ 0    ", dict(non_negative=False))]:
    model = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, **kw)
    x     = model.x.ravel()
    print(f"{label}: x = {np.round(x, 4)}, s = {model.y[0, 0]:7.4f}, "
          f"rho = {model.rho:.4f}")
    print(f"{'':24}A_ub @ x = {(A_ub @ x)[0]:.4f}, A_eq @ x = {np.round(A_eq @ x, 4)}")
```

Output:

```text
x ≥ 0 (default)       : x = [0.1667 2.1667 0.    ], s =  0.0000, rho = 0.5774
                        A_ub @ x = 2.3333, A_eq @ x = [0.1667 2.1667]
x2 ≤ 2, x3 = 0.25     : x = [0.25 2.   0.25], s =  0.0000, rho = 0.8660
                        A_ub @ x = 2.5000, A_eq @ x = [0. 2.]
x ≥ 0, free slack     : x = [0.5 2.5 0. ], s = -1.0000, rho = 0.0000
                        A_ub @ x = 3.0000, A_eq @ x = [0.5 2.5]
x unbounded, s ≥ 0    : x = [-0.   2.5 -0.5], s =  0.0000, rho = 0.0000
                        A_ub @ x = 2.0000, A_eq @ x = [0.5 2.5]
```

With x ≥ 0, the three rows cannot all be met (x2 = 2.5 and x1 ≥ 0.5 exceed the sum of 2), and the misfit is shared by the rows (ρ > 0). The bounds are always met exactly: an upper bound x2 ≤ 2 and a fixed value x3 = 0.25 (lo = hi) restrict the solution further and increase ρ. With `slack_sign=False`, the slack of the inequality row is free (s < 0), so the row imposes no restriction and the equalities are met exactly. Without the default non-negativity (`non_negative=False`), the constraints are feasible with x3 < 0 (the printed −0. is a rounded zero).

### 4. Step 2 and α

```python
import numpy  as     np
from   lppinv import lppinv

# Step 2: α selects among the equally well-fitting solutions (x ≥ 0, inconsistent rows)
# x = vec(X), X ∈ ℝ^{3×3}: row sums [1, 10, 10] and column sums [1, 10, 11] (21 vs 22),
# trace(X) ≤ 5

A_eq   = np.vstack([np.kron(np.eye(3), np.ones((1, 3))),     # row sums
                    np.kron(np.ones((1, 3)), np.eye(3))])    # column sums
b_eq   = np.array([1.0, 10.0, 10.0, 1.0, 10.0, 11.0])
A_ub   = np.eye(3).reshape(1, -1)                            # trace
b_ub   = np.array([5.0])
for alpha in (0.0, 0.5, 1.0):                                # Lasso, Elastic Net, Ridge
    model = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, alpha=alpha)
    print(f"alpha = {alpha}: rho = {model.rho:.4f}, residual = {model.residual:.4f}, "
          f"solver = {model.status['solver']}, final = {model.status['final']}")
    print(np.round(model.x.reshape(3, 3), 4))
print("trace slack s =", round(float(model.y[0, 0]), 4))
try:
    lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, alpha=[0.0, 1.0])
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

Output:

```text
alpha = 0.0: rho = 0.4082, residual = 0.4082, solver = ECOS, final = optimal
[[0.     0.5421 0.6245]
 [0.412  2.273  7.4817]
 [0.4214 7.0183 2.727 ]]
alpha = 0.5: rho = 0.4082, residual = 0.4082, solver = CLARABEL, final = optimal
[[0.     0.4167 0.75  ]
 [0.4167 2.3333 7.4167]
 [0.4167 7.0833 2.6667]]
alpha = 1.0: rho = 0.4082, residual = 0.4082, solver = OSQP, final = optimal
[[0.     0.4167 0.75  ]
 [0.4167 2.3333 7.4167]
 [0.4167 7.0833 2.6667]]
trace slack s = 0.0
LPPinvInputError: alpha must be a single value in [0, 1]; selection by NRMSE was removed in 5.0.0.
```

The row and column sums are inconsistent, so the smallest achievable residual under x ≥ 0 and trace(X) ≤ 5 is ρ = 0.4082, attained for every α (`residual` = `rho`). α only selects among these solutions by their L1 and/or L2 distance to the unbounded anchor `zhat0`, which is why the Lasso solution (α = 0) differs from the others; for α = 0, several minimizers may exist, and the solver returns one of them. α must be a single value in [0, 1]: lists of candidates, which were evaluated by NRMSE before version 5.0.0, raise an *LPPinvInputError*.

### 5. Reading the Results

```python
import numpy  as     np
from   lppinv import lppinv

# Reading the results: status, Step 1 vs Step 2, summary, and a bootstrap t-test

seed   = 123456789
rng    = np.random.default_rng(seed)
A_ub   = np.abs(rng.normal(size=(10, 40)))                   # A_ub @ x ≥ 0 for x ≥ 0
b_ub   = rng.uniform(1.0, 2.0, size=10)
A_eq   = rng.normal(size=(5, 40))
b_eq   = A_eq @ rng.uniform(0.0, 1.0, size=40)               # consistent with x ≥ 0
b_ub[0] = -1.0                                               # infeasible first row
model  = lppinv(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

# results
for key, val in model.status.items():
    print(f"{key:12}: {val}")
print("rho =", f"{model.rho:.6f}", " residual =", f"{model.residual:.6f}")
print("||zhat|| (Step 1) =", f"{np.linalg.norm(model.zhat):.6f}",
      " ||z|| (Step 2) =", f"{np.linalg.norm(model.z):.6f}")
print("violation of the first row:", f"{(A_ub[0] @ model.x.ravel() - b_ub[0]):.6f}")

model.summary(display=True)

print("  Bootstrap t-test:")
for kw, val in model.ttest(sample_size=30, seed=seed).items():
    print(f"    {kw}: {float(val):.6f}")
```

Output:

```text
step1       : converged
rho         : bvls
final       : optimal
solver      : OSQP
retry       : None
polish      : True
tol_v       : 0.0008640490391084263
kkt_error   : 1.3615129212596414e-14
scale       : False
bound_error : 0.0
eq_error    : 2.8108601893111292e-15
rho = 2.528664  residual = 2.528664
||zhat|| (Step 1) = 2.569309  ||z|| (Step 2) = 2.569309
violation of the first row: 1.647750
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                 [0, inf]
  Slack sign (S ≥ 0):             True
  Column scaling:                False
  Fixed variables:                  40

Numerical Stability:
  kappaC:                       7.5963
  kappaB:                       1.0000
  kappaA:                       7.5963
  rank:                             15

Goodness of Fit:
  r2_partial:                      nan
  nrmse:                        0.4163
  nrmse_partial:                   nan
  z_lower:        min=   -2.6072  max= 0.000e+00  mean=   -0.2272  sd=    0.5857
  z_upper:        min= 0.000e+00  max=    5.6499  mean=    0.4923  sd=    1.2692
  x_lower:        min=   -1.9523  max= 0.000e+00  mean=   -0.0894  sd=    0.3382
  x_upper:        min= 0.000e+00  max=    4.2307  mean=    0.1937  sd=    0.7329
  y_lower:        min=   -2.6072  max= 0.000e+00  mean=   -0.7784  sd=    0.9746
  y_upper:        min= 0.000e+00  max=    5.6499  mean=    1.6868  sd=    2.1119
  Bootstrap t-test:
    p_one_left: 0.184626
    p_one_right: 0.815374
    p_two_sided: 0.369252
    nrmse: 0.416338
    mean_null: 0.429501
    std_null: 0.079046
    failures: 0.000000
```

`status` reports the outcome of the bounded first step (`"step1"`), the source of w* (`"rho"`), the outcome and solver of the second step (`"final"`, `"solver"`, `"retry"`, `"polish"`), the verification tolerance (`"tol_v"`) and the achieved errors (`"kkt_error"`, `"bound_error"`, `"eq_error"`); it is the reliable way to check a fit programmatically. Since A_ub @ x ≥ 0 for x ≥ 0, the first row (≤ −1) cannot be met; it is violated in least squares, while the other rows are traded off against it, and ρ > 0. The bootstrap t-test compares the observed NRMSE with the mean of NRMSE values from resampled residuals; see the pyclsp README for its interpretation.

## User Reference

For comprehensive information on the estimator's capabilities, advanced configuration options, and implementation details, please refer to the [pyclsp module](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming"), on which LPPinv is based, and to the docstring of `lppinv()`.

To ensure cross-platform reproducibility, all CLSP implementations use a modified condition number function based on singular values, with a relative cutoff equal to `cond_tolerance * the largest singular value`.

```python
lppinv(c, A_ub, b_ub, A_eq, b_eq, non_negative, bounds, replace_value, tolerance, cond_tolerance, slack_sign, **kwargs)
```

**LPPINV Parameters:**

`c` : *array_like* of shape *(p,)*, optional<br>
Objective function coefficients. Accepted for API parity with `scipy.optimize.linprog`; not used by CLSP (the objective is not optimized).

`A_ub` : *array_like* of shape *(i, p)*, optional<br>
Matrix for inequality constraints `A_ub @ x <= b_ub`. Each row receives a slack variable s_i (S_ii = +1).

`b_ub` : *array_like* of shape *(i,)*, optional<br>
Right-hand side vector for inequality constraints. Rows with +inf impose no restriction and are dropped; −inf raises an *LPPinvInputError* (no x satisfies `A_ub[i] @ x <= -inf`); NaN values raise a *CLSPError*.

`A_eq` : *array_like* of shape *(j, p)*, optional<br>
Matrix for equality constraints `A_eq @ x = b_eq`.

`b_eq` : *array_like* of shape *(j,)*, optional<br>
Right-hand side vector for equality constraints. ±inf raises an *LPPinvInputError*; NaN values raise a *CLSPError*.

`non_negative` : *bool*, default = *True*<br>
If *True*, a missing (*None*) lower bound is replaced by 0, and negative bounds raise an *LPPinvInputError*. If *False*, no default nonnegativity is applied.

`bounds` : *sequence* of *(low, high)*, optional<br>
Bounds on variables, passed to CLSP as `bounds` and enforced exactly in both steps (bounded-variable least squares in Step 1, constraints in Step 2). If a single tuple *(low, high)* is given, it is applied to all variables; a list (or tuple) of *(low, high)* pairs gives one pair per variable (or a single pair). *None* (or ±*np.inf*) denotes an unbounded side; low = high fixes a variable; low ≤ high is required. If *None*, defaults to *(0, None)* for each variable (non-negativity) or, with `non_negative=False`, to no bounds.

`replace_value` : *float* or *None*, default = *np.nan*<br>
Final replacement value for any cell in the returned CLSP.x that violates the specified bounds by more than the given tolerance. Since version 5.0.0, the bounds are enforced exactly by CLSP, so this is only a safeguard (`model.status["bound_error"]` reports the largest bound violation, 0 in practice).

`tolerance` : *float*, default = *square root of machine epsilon*<br>
Tolerance of CLSP (passed as `tolerance`: the bound and KKT tolerances of the bounded first step and the verification tolerance of the second step, tol_v = sqrt(`tolerance`) · max(1, ‖`b`‖₂)) and of the bounds in `replace_value`.

`cond_tolerance` : *float* or *None*, default = *None*<br>
Singular-value cutoff for the custom condition number function. If *None*, the implementation uses an internal relative cutoff of `1e-14`.

`slack_sign` : *bool*, default = *True*<br>
If *True*, the slacks of the inequality rows must be ≥ 0 (pyclsp's `slack_sign`), so that each row `A_ub[i] @ x <= b_ub[i]` is free on its feasible side and a least-squares row on its infeasible side. If *False*, the slacks are free and the inequality rows impose no restriction (Example 3).

Please note that either `A_ub` and `b_ub` or `A_eq` and `b_eq` must be provided.

**CLSP Parameters** (passed via `**kwargs` to `CLSP.solve()`; see the pyclsp README):<br>
`r` : *int*, default = *1*<br>
Number of refinement iterations. Since LPPinv has no model rows (`M`), the refinement has no effect (`model.r` = 1).

`Z` : *np.ndarray* or *None*, default = *None*<br>
A symmetric idempotent matrix (projector) on z = (x, s) for the Bott–Duffin inverse, of size (p + i) × (p + i) (after dropping the rows with `b_ub` = +inf). With bounds or signed slacks, `Z` must be diagonal (0/1), and the coordinates excluded by `Z` must admit 0.

`rcond` : *float* or *bool*, default = *False*<br>
Relative cutoff on the singular values of `A`·`Z`, which defines `A`_k (used in both steps). *False*: sqrt(max(`A`.shape) · eps); *True*: `tolerance`; a float: the given cutoff. A discarded singular value is treated as an exact zero.

`iteration_limit` : *int*, default = *50*<br>
Maximum number of passes of the bounded first step, after which scipy's BVLS is used.

`final` : *bool*, default = *True*<br>
If *True*, the second step is solved: z minimizes a weighted L1/L2 norm around the unbounded first-step estimate `zhat0` subject to `A`_k`z` = w* and to the bounds and slack signs, where w* = `A`_k`zhat` is the fitted vector of the bounded least-squares problem of the first step. If *False*, `z` = `zhat`.

`alpha` : *float* or *None*, default = *None*<br>
    Regularization parameter (weight) in the final convex program, a single value in [0, 1]:<br>
    - `α = 0`: Lasso (L1 norm)<br>
    - `α = 1`: Tikhonov Regularization/Ridge (L2 norm)<br>
    - `0 < α < 1`: Elastic Net<br>
    If *None*, α = 1: the solution is then unique and, for `Z` = I, the minimum-norm point (x, s) among the bounded least-squares solutions. A *list*, *tuple* or *np.ndarray* raises an *LPPinvInputError* (the selection by NRMSE was removed in version 5.0.0, since all α attain the same residual); NaN, inf, non-numeric and out-of-range values raise a *CLSPError*.

`scale` : *bool*, default = *False*<br>
If *True*, the columns of `A`·`Z` are equilibrated in both steps (the minimum-norm selection is then made in the scaled variables). Requires a diagonal (0/1) `Z`.

`solver`, other keyword arguments : optional<br>
CVXPY solver (default: ECOS for α = 0, OSQP for α = 1 with a CLARABEL retry, CLARABEL for 0 < α < 1) and solver options for the second step.

The keyword arguments `problem`, `C`, `S`, `M` and `b` are ignored, since LPPinv builds the canonical form itself.

**Returns:**
*CLSP*<br>
The fitted CLSP object (see the pyclsp README for all attributes and methods). Selected attributes:

`model.x`             : *np.ndarray*<br>
`p × 1` vector of the variables x.

`model.y`             : *np.ndarray*<br>
vector of the slacks s of the inequality rows (s = `b_ub` − `A_ub` @ x up to the misfit of the rows; empty without `A_ub`).

`model.z`, `model.zhat`, `model.zhat0` : *np.ndarray*<br>
final solution z = (x, s), first-step (bounded least-squares) estimate, and unbounded first-step estimate `A`_k⁺`b` (the anchor of the second step).

`model.A`, `model.b`  : *np.ndarray*<br>
canonical matrix [`A_ub` | I; `A_eq` | 0] and right-hand side [`b_ub`; `b_eq`] (without the dropped rows).

`model.x_lb`, `model.x_ub`, `model.bounds`, `model.slack_sign` : *np.ndarray*, *list*, *bool*<br>
bounds on x as applied (-inf/inf if unbounded), as passed to CLSP, and the sign restriction of the slacks.

`model.rho`, `model.residual` : *float*<br>
smallest residual norm ‖`b` − w*‖₂ under the bounds (0 for feasible constraints; *None* if `final` is *False*) and residual norm ‖`b` − `A``z`‖₂ of the returned z.

`model.status`        : *dict*<br>
statuses of the last solve: `"step1"`, `"rho"`, `"final"`, `"solver"`, `"retry"`, `"tol_v"`, `"kkt_error"`, `"bound_error"`, `"eq_error"`, `"scale"` (Example 5).

`model.fixed`, `model.fixed_at`, `model.rank`, `model.sv_discarded` : *np.ndarray*, *int*<br>
components of z at a bound at the end of the first step and the pass of fixing, effective rank of `A`·`Z`, and discarded singular values.

`model.kappaC`, `model.kappaB`, `model.kappaA` : *float*<br>
spectral κ() for the constraint rows, for *B* = `A`*C_canon*⁺ and for `A` (without model rows, `A` consists of the constraint rows only).

`model.nrmse`, `model.z_lower`, `model.z_upper`, `model.x_lower`, `model.x_upper`, `model.y_lower`, `model.y_upper` : *float*, *np.ndarray*<br>
NRMSE of the constraint rows and heuristic condition-weighted sensitivity bands (`nrmse_partial` and `r2_partial` are *nan* without model rows).

The methods `model.summary()`, `model.corr()` (row-deletion diagnostics of the constraint rows) and `model.ttest()` are available. `model.cv_alpha()` is not informative for LPPinv, since it holds out model rows, which LPPinv does not have; it returns α = 1 with a *RuntimeWarning*.

**Errors:**<br>
An *LPPinvInputError* (`lppinv.lppinv.LPPinvInputError`) is raised if no complete constraint set (`A_ub`, `b_ub`) or (`A_eq`, `b_eq`) is given; if a matrix and its right-hand side have different numbers of rows, or `A_ub` and `A_eq` different numbers of columns; if the number of bounds is neither 1 nor the number of variables; for negative bounds with `non_negative=True`; for `b_ub` = −inf or an infinite `b_eq`; if no row with a finite right-hand side remains; and for a list, tuple or array of α. Errors detected by pyclsp (e.g., low > high in `bounds`, NaN entries, an invalid scalar α or `Z`) raise a *CLSPError* (`clsp.errors.CLSPError`).

## Warnings and Troubleshooting

LPPinv issues no warnings of its own: the *RuntimeWarning*s of a fit come from pyclsp, i.e., from a bounded first step that did not converge (scipy's BVLS is then used, `status["step1"]` = *'bvls fallback'*, or the last feasible point is kept) or from a second step that fell back to `zhat` (`status["final"]` = *'fallback: <reason>'*). See the section *Warnings and Troubleshooting* of the [pyclsp README](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") for their meaning and remedies (e.g., `scale=True` for columns of very different magnitude, another `solver`, a larger `iteration_limit` or `rcond`). The reliable way to check a fit programmatically is `model.status` (Example 5); `warnings.simplefilter("error", RuntimeWarning)` turns the warnings into exceptions.

## Limitations

- **The objective is not optimized.** `c` is accepted for API parity with `scipy.optimize.linprog` but is not used: for feasible constraints, LPPinv returns the minimum-norm feasible point (α = 1) or the feasible point closest to the anchor `zhat0` in the L1/L2 sense (α < 1), not an optimal vertex of the LP (Example 1). For the LP optimum, use an LP solver.
- **Soft constraint rows.** The inequality and equality rows are least-squares rows: for infeasible constraints, they are traded off against each other and are not met exactly (Example 2); a hard-constraint mode is not implemented in pyclsp. Only the bounds on x and the signs of the slacks are enforced exactly. A large row weight approximates a hard row at the cost of a larger κ(`A`).
- **The norm includes the slacks.** The minimum-norm selection of Step 2 (α = 1) minimizes ‖x‖² + ‖s‖², not ‖x‖² alone, so the result depends on how the inequality rows are scaled (with `scale=True`, it is made in the scaled variables).
- **Numerical range.** All steps use `A`_k, so a singular value below the cutoff `rcond` is treated as an exact zero, and ρ is the smallest residual only within the numerical range of `A`.
- **Accuracy of Step 2.** The second step is verified to the tolerance tol_v = sqrt(`tolerance`) · max(1, ‖`b`‖₂) (feasibility, not optimality); values near an active bound may differ from the exact minimizer of Step 2 by a small multiple of the solver tolerance.
- **No model rows.** LPPinv has no model rows, so `r` > 1, `nrmse_partial`, `r2_partial`, `ttest(partial=True)` and `cv_alpha()` have no effect or are not informative.
- **Dense linear algebra.** All matrices are dense, every solve computes an SVD of the (i + j) × (p + i) canonical matrix and passes it to CVXPY, so time and memory grow quickly with the number of variables (see the pyclsp README for indicative timings).

## What's New in 5.0.0

Version 5.0.0 adapts LPPinv to pyclsp 5.0.0; the full list of changes is in [CHANGELOG.md](CHANGELOG.md).

- Bounds and non-negativity are passed to pyclsp as native `bounds` and enforced exactly in both steps, instead of additional bound rows with free slacks and a replacement of out-of-bound values by `replace_value` (now only a safeguard).
- The slacks of the inequality rows are signed (`slack_sign=True`, new parameter), so that `A_ub` @ x ≤ `b_ub` is a one-sided restriction.
- The canonical matrix contains only the rows of `A_ub` and `A_eq`; ρ, `residual` and `status` of pyclsp report the infeasibility of the constraints and the outcome of both steps.
- `alpha` is a single value in [0, 1] (*None* = 1); the pyclsp options `rcond`, `iteration_limit`, `final`, `scale` and `solver` are passed through.
- Only rows with `b_ub` = +inf are dropped as restrictions without effect; `b_ub` = −inf and an infinite `b_eq` raise an *LPPinvInputError* instead of disappearing.
- A test suite, CI workflows, the full MIT license text and a changelog; Python ≥ 3.11.

**Breaking changes:** Python ≥ 3.11 and pyclsp ≥ 5.0.0; different estimates (native bounds, signed slacks, no bound rows; `A`, `b`, `y` and the diagnostics refer to the constraint rows only); α lists raise an *LPPinvInputError*; NaN right-hand sides raise a *CLSPError* and `b_ub` = −inf or an infinite `b_eq` an *LPPinvInputError* instead of being dropped; `slack_sign` is a new positional parameter after `cond_tolerance`. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use LPPinv in your work, please cite the method:

Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

and the version of the package you used (`lppinv.__version__`):

pylppinv, version 5.0.0: Linear Programming via Regularized Least Squares [Python package]. [https://github.com/econcz/pylppinv](https://github.com/econcz/pylppinv)

```bibtex
@article{bolotov2025clsp,
  author  = {Bolotov, I.},
  title   = {{CLSP}: Linear Algebra Foundations of a Modular Two-Step Convex
             Optimization-Based Estimator for Ill-Posed Problems},
  journal = {Mathematics},
  year    = {2025},
  volume  = {13},
  number  = {21},
  pages   = {3476},
  doi     = {10.3390/math13213476}
}

@software{pylppinv,
  title   = {pylppinv: Linear Programming via Regularized Least Squares},
  version = {5.0.0},
  url     = {https://github.com/econcz/pylppinv}
}
```

## Acknowledgements

The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

## Bibliography
Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

## License

MIT License — see the [LICENSE](LICENSE) file.
