# Validation

Checked locally on 2026-11-20 using Python 3.13.2 on macOS.

| Check | Result |
|---|---|
| `python -m pytest -q` | 56 passed |
| `python demo.py` | Completed; prior 0.150000, posterior 0.334636 |
| Lecture query, `Rain?` given `Wet Grass?=true`, min-degree | P(Rain?=True) = 0.842685 |
| `python -m pip check` | No broken requirements |
| `git diff --check` | Passed |
| Fresh environment with runtime, test and optional plotting dependencies | Installed successfully |
| Tests run from a separate copy of the release files | 56 passed |
| Both documented CLI queries | Completed with expected probabilities |
| Optional network plotting with a noninteractive backend | Rendered successfully |

The exact installed packages are recorded in `requirements-tested.txt` at the repository root. To recreate this local test environment, use Python 3.13 and install that file. The normal setup uses the smaller runtime requirements; tests additionally need `requirements-dev.txt`.

## Correctness checks

- Posterior probabilities and evidence probabilities agree with independent exhaustive enumeration on all three complete supplied networks, under both elimination heuristics.
- Joint queries on eight seeded generated networks agree with enumeration for min-fill, min-degree and an explicit elimination order.
- A hand-calculated posterior and explicit local CPT entries check XML outcome and parent ordering independently of the inference comparison.
- Graph tests cover chains, forks, colliders, an observed grandchild of a collider, disconnected variables and empty endpoint sets.
- A constructed example distinguishes MAP over a subset from the projection of MPE. Additional checks cover conditional explanation probabilities, impossible evidence, scalar factors and repeated queries without mutation.
- Input checks cover invalid networks, cycles, duplicate assignments, unknown variables, malformed XML and incomplete tables. The archived hiring draft is expected to be rejected.
- The demo runs from a different working directory, and importing modules produces no output or file access to example paths.

## What has not been tested

The included CI workflow targets Python 3.11 and 3.13 but has not run on GitHub for this local copy. Only Python 3.13.2 was exercised locally. Plot rendering was checked with a noninteractive backend; opening an interactive window was not tested. There are no scalability benchmarks or claims about suitability for real-world decisions.

The original GitHub repository remained at `a9c1aeae408204bd70bbeb8e02702c2cdd448f39` after this work.
