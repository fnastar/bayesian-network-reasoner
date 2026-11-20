# Bayesian network reasoner

A Python implementation of exact inference for small Boolean Bayesian networks. It calculates how observations change probabilities, checks conditional independence and finds likely explanations.

Based on my KR21 coursework, with repaired inference routines, XMLBIF loading, runnable examples and tests against exhaustive enumeration.

## Try it

From the project directory, using Python 3.11 or 3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
```

On Windows, activate with `.venv\Scripts\activate`.

The bundled teaching example relates a family's absence, a light, a dog being outside and hearing barking. It uses specified probabilities, not learned data.

```text
P(family out)                 = 0.150000
P(family out | hear barking)  = 0.334636
```

Hearing barking raises the estimated probability that the family is out from 15% to about 33%. It does not establish why the dog is barking.

## What it implements

| Capability | Approach |
|---|---|
| Posterior probabilities | Reduce factors by evidence, multiply compatible rows, sum out hidden variables, then normalise |
| Elimination order | Min-degree and min-fill heuristics, recomputed after each elimination |
| Conditional independence | D-separation through the ancestral moral graph |
| MAP | Sum over hidden variables, then select the most likely joint assignment of the query variables |
| MPE | Select the most likely assignment of all unobserved variables |

Factor operations use pandas; graph operations use NetworkX. The inference algorithms are implemented in `BNReasoner.py`.

## Example query

```bash
python demo.py --query family-out --evidence hear-bark=true
python demo.py --network testing/lecture_example.BIFXML --query "Rain?" --evidence "Wet Grass?=true" --order min-degree
```

Or use Python from the project directory:

```python
from BNReasoner import BNReasoner

reasoner = BNReasoner("testing/dog_problem.BIFXML")
print(reasoner.query(["family-out"], {"hear-bark": True}))
print(reasoner.map(["family-out"], {"hear-bark": True}))
print(reasoner.mpe({"hear-bark": True}))

# Original coursework argument order: X, Z (observed), Y.
print(reasoner.d_separated({"light-on"}, {"family-out"}, {"hear-bark"}))
```

Queries return a DataFrame with Boolean assignment columns and a `p` column. MAP/MPE return an `Explanation` with an assignment and its probability conditional on the evidence. Evidence must use Python `True` or `False`; unknown variables and impossible evidence raise `ValueError`. Queries do not modify the network.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests compare both elimination heuristics with an independent full-world calculation on the three supplied teaching networks and eight seeded generated networks. They also cover XML outcome ordering, a hand-calculated posterior, chains/forks/colliders, observed collider descendants, disconnected graphs, zero-probability evidence, and the distinction between MAP and MPE.

A GitHub Actions workflow is included for Python 3.11 and 3.13. See [validation notes](docs/validation.md) for the checks actually run locally.

Optional network plotting uses `BayesNet.draw_structure()` and requires `python -m pip install -r requirements-plot.txt`.

## Limits

- Boolean chance variables only; no continuous variables, parameter learning or causal-effect estimation.
- Exact inference can require exponential time and memory. MAP/MPE materialise the query distribution and are intended for small teaching networks, not large deployments.
- Computation uses ordinary floating-point probabilities, so very small probabilities can underflow.
- Example probabilities are supplied assumptions. They are not empirical findings or evidence of real-world decision quality.

