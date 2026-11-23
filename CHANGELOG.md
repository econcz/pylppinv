# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [5.0.0] – 2026-11-23

Version 5.0.0 adapts LPPinv to pyclsp 5.0.0. The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

### Added

- `slack_sign` parameter of `lppinv()` (default *True*): the slacks of the inequality rows (`A_ub` @ x + s = `b_ub`) must be ≥ 0, passed to pyclsp as `slack_sign`, so that each row `A_ub[i] @ x <= b_ub[i]` is free on its feasible side and a least-squares row on its infeasible side; *False* leaves the slacks free.
- Pass-through of the pyclsp 5.0.0 options of `CLSP.solve()` via `**kwargs`, documented in the docstring and the README: `r`, `Z`, `rcond`, `iteration_limit`, `final`, `alpha` (a single value), `scale`, `solver` and CVXPY solver options. The new pyclsp attributes (`zhat0`, `rho`, `residual`, `status`, `fixed`, `fixed_at`, `rank`, `sv_discarded`, `x_lb`, `x_ub`) are available on the returned object.
- *LPPinvInputError* for a list, tuple or array of `alpha` ("alpha must be a single value in [0, 1]; selection by NRMSE was removed in 5.0.0.") and for a problem without any row with a finite right-hand side.
- Test suite `tests/test_lppinv.py` (feasible and infeasible LP-type problems, equality-only problems, bounds and slack signs, `alpha` validation, a comparison of the bounded first step with `scipy.optimize.lsq_linear(..., method="bvls")` on the same canonical system, pass-through of the options; runs under pytest or as a script, against the installed package with `LPPINV_TEST_INSTALLED=1`, with a pyclsp source tree prepended by `CLSP_SRC=<path>`).
- CI: a `test` job (Python 3.11 and 3.12) in the PyPI workflow, on which the release build depends, and a `tests.yml` workflow (push and pull request; Python 3.11, 3.12 and 3.13) that builds the wheel and runs the tests against the installed wheel. Both require pyclsp 5.0.0 on PyPI.
- Full MIT license text in `LICENSE` (previously only the title line).
- `CHANGELOG.md`; README sections on how LPPinv works (canonical form, Step 1, Step 2), six runnable examples with their output, an extended parameter and attribute reference, warnings and troubleshooting, limitations, and citation.

### Changed

- Bounds on x (including the default non-negativity) are passed to pyclsp as native `bounds` and enforced exactly in both steps (bounded-variable least squares in Step 1, constraints in Step 2), instead of 2·p additional constraint rows (x + s = ub, x − s = lb) with free slacks.
- The canonical form contains only the rows of `A_ub` and `A_eq`: `A` = [`A_ub` | I; `A_eq` | 0], `b` = [`b_ub`; `b_eq`], and `model.y` holds only the slacks of the inequality rows.
- A tuple `bounds` is read as a single (low, high) pair only if it has two entries that are *None* or real numbers; a missing lower bound is replaced by 0 if `non_negative` is *True* before the bounds are passed to pyclsp.
- `replace_value` is a safeguard only: it is applied to the cells of x outside `model.x_lb`/`model.x_ub` by more than `tolerance`, which no longer occurs, since the bounds are enforced exactly.
- `tolerance` is passed to pyclsp, where it also sets the bound and KKT tolerances of the bounded first step and the verification tolerance of the second step.
- Only rows with `b_ub` = +inf are dropped (with their slacks), as restrictions without effect; NaN values are passed to pyclsp, which raises a *CLSPError*. Before 5.0.0, the rows with ±inf served as the absent sides of the bound rows; with native bounds, this reason no longer applies.
- The keyword arguments `problem` and `M` are ignored (as `C`, `S` and `b`), since LPPinv builds the canonical form itself.
- Docstring and README state that the rows are least-squares rows (only the bounds and the signs of the slacks are hard), that the objective `c` is not optimized, and that the default result (α = 1) is the minimum-norm point (x, s) among the bounded least-squares solutions.
- Requires Python ≥ 3.11 and pyclsp ≥ 5.0.0; classifiers for Python 3.11, 3.12 and 3.13.
- Version 5.0.0 in `pyproject.toml` and `lppinv/__init__.py`.

### Fixed

- The slacks of the inequality rows were unrestricted, so that `A_ub` @ x ≤ `b_ub` was not a one-sided restriction.
- Bounds were only soft rows, so the estimate could violate them and the violating cells were replaced by `replace_value` (NaN by default) in `model.x` (but not in `model.z`).
- A tuple of (low, high) pairs, e.g. `((0, 1), (0, 2))`, was replicated as a single pair and raised an unrelated *TypeError*.
- NaN right-hand sides were dropped silently together with their rows.
- A problem in which all right-hand sides were infinite was passed to pyclsp without rows.
- Rows with `b_ub` = −inf (no x satisfies them) and equality rows with an infinite `b_eq` were dropped as if absent, so that an impossible constraint disappeared and the remaining rows could report a zero residual; they now raise an *LPPinvInputError*.
- A `problem` keyword argument raised an unrelated *TypeError* (multiple values for `problem`).

### Removed

- Support for Python 3.10.
- The bound rows of the canonical form and the bounds in the right-hand side `b`.
- The selection of α by NRMSE (lists of candidates, the error rule for `alpha=None`), following pyclsp 5.0.0.

### Breaking

- Python ≥ 3.11 and pyclsp ≥ 5.0.0 are required.
- Different estimates for the same inputs: native bounds, signed inequality slacks and the absence of bound rows change x, and `A`, `b`, `y`, `C_idx`, `S_idx`, the condition numbers, NRMSE and the sensitivity bands refer to the rows of `A_ub` and `A_eq` only. The pyclsp 5.0.0 changes (Step 1 by one SVD with the cutoff `rcond` on the singular values of `A`·`Z`, Step 2 subject to `A`_k`z` = w* anchored at `zhat0`) apply as well.
- `alpha` given as a list, tuple or array raises an *LPPinvInputError*; `alpha=None` means α = 1.
- NaN right-hand sides raise a *CLSPError* instead of being dropped; `b_ub` = −inf and an infinite `b_eq` raise an *LPPinvInputError*.
- `slack_sign` is a new parameter after `cond_tolerance`: an eleventh positional argument binds to `slack_sign` instead of `*args`.

## [4.0.0] – 2026-06-06

- Bug fixes.

## [3.5.0] – 2026-04-22

- Cross-platform condition-number tolerance in CLSP.

## [1.6.0] – 2025-12-21

- GitHub Actions workflow for PyPI publishing.

## [1.4.0] – 2025-11-06

- Bug fixes and refreshed README.

## [1.3.0] – 2025-09-17

- `non_negative` flag and replacement of out-of-bound values.

## [1.0.4] – 2025-08-19

- Complete redesign based on pyclsp.

## [0.4.2] – 2024-02-17

- Bug fix.

## [0.4.0] – 2024-02-17

- Major update.

## [0.3.8] – 2022-02-06

- Bug fix in the calculation of the NRMSE.

## [0.3.7] – 2022-02-05

- MSE replaced by RMSE.

## [0.3.6] – 2021-06-03

- Terminology update.

## [0.3.2] – 2021-05-28

- Update of the estimator and the README.

## [0.2.1] – 2021-05-28

- Package metadata.

## [0.1.0d] – 2021-05-28

- Initial commit.
