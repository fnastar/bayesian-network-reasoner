# Provenance and changes

## Source

This copy starts from [fnastar/KR21_project2](https://github.com/fnastar/KR21_project2) at revision [a9c1aeae408204bd70bbeb8e02702c2cdd448f39](https://github.com/fnastar/KR21_project2/tree/a9c1aeae408204bd70bbeb8e02702c2cdd448f39). It is maintained separately; no changes were made to the source repository.

The source revision remains available in the original repository. Machine-specific editor files and `.DS_Store` are excluded from this copy. The public release is prepared from the reviewed files without importing the original development history.

## Credits

- Felix Nastar maintains this version.
- The original project was KR21 Group 32 coursework.
- The original README identifies `BayesNet.py` as course-provided support code. This version adapts that container and replaces its loader.
- The lecture XMLBIF examples credit Andreas Sauter.
- The dog-problem file retains its JavaBayes and XMLBIF source comments.
- The archived hiring example credits Group 32.

The original repository has no licence file. This copy does not assign a new licence to inherited course material.

## Repairs in this version

- Removed file access and demonstration code at import time, including absolute paths to files absent from the repository.
- Replaced the incomplete marginalisation path with factor-based variable elimination, conditional normalisation and deterministic min-degree/min-fill orders.
- Replaced path-by-path d-separation with ancestral moralisation, which handles disconnected nodes and observed descendants of colliders.
- Added working MAP/MPE interfaces. The previous MPE draft imported a missing `Ordering` module. The current explanation methods are deliberately simple and materialise the posterior for small query sets.
- Replaced the XMLBIF loader's implicit Boolean mapping with explicit `OUTCOME` labels and declared parent order. The old code constructed rows in False/True order even when the XML declared True first. Results from this version should not be described as the original submission's results.
- Added validation for malformed probability tables, invalid evidence and impossible conditioning events.
- Kept pandas and NetworkX as the core dependencies; plotting is optional. Removed the pgmpy dependency.
- Added a portable demo, a concise project README, independent correctness tests and a CI workflow.

The XML loader follows [XMLBIF 0.3](https://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/): parent states follow declared GIVEN order, the child varies fastest, and state order follows OUTCOME declarations.

## Archived draft

`archive/hiring-draft.BIFXML` preserves the original hiring example without inventing missing probabilities. It declares `Communication skills` but defines `Communication Skills`, and `Decision making ability` has no definition. The loader rejects it. It is not a demonstrated application of this project.

## API changes

The maintained API is `query`, `evidence_probability`, `elimination_order`, `d_separated`, `map` and `mpe` on `BNReasoner`. The original `marginal_distribution(Q, E, order)` entry point remains as an adapter; fixed query values filter the posterior without renormalising it.

The original d-separation order `d_separated(X, Z, Y)` is retained. Constructors accept either a file path or a `BayesNet` object. Incomplete internal helpers from the coursework draft are not preserved as public APIs.

`MAP().run(bn, query, evidence=None, order="min-fill")` and `MPE().run(bn, evidence=None)` return an `Explanation`. This replaces the old partial MAP helpers and the old MPE return table. Ties select the first assignment after sorting variables and ordering False before True.
