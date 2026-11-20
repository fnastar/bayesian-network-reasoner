"""Exact inference for small Boolean Bayesian networks."""

from dataclasses import dataclass
from itertools import combinations
from os import PathLike

import networkx as nx
import pandas as pd

from BayesNet import BayesNet


@dataclass(frozen=True)
class Explanation:
    """A most likely assignment and its probability conditional on the evidence."""

    assignment: dict[str, bool]
    probability: float


def multiply_factors(factors):
    """Multiply compatible rows, including disjoint and scalar factors."""
    result = pd.DataFrame({"p": [1.0]})
    for factor in factors:
        shared = [c for c in result.columns if c != "p" and c in factor.columns]
        if shared:
            result = result.merge(factor, on=shared, how="inner")
        else:
            result = result.merge(factor, how="cross")
        result["p"] = result.pop("p_x") * result.pop("p_y")
    return result


def sum_out(factor, variable):
    """Marginalise one variable; a zero-variable factor is a single number."""
    remaining = [c for c in factor.columns if c not in (variable, "p")]
    if variable not in factor:
        raise ValueError(f"Variable {variable!r} is not in the factor")
    if not remaining:
        return pd.DataFrame({"p": [factor["p"].sum()]})
    return factor.groupby(remaining, as_index=False, sort=True)["p"].sum()


class BNReasoner:
    def __init__(self, net: str | PathLike | BayesNet):
        if isinstance(net, (str, PathLike)):
            net_path = net
            net = BayesNet()
            net.load_from_bifxml(net_path)
        if not isinstance(net, BayesNet):
            raise TypeError("Expected a BIFXML path or BayesNet object")
        net.validate()
        self.bn = net
        self.G = net.structure
        self.variables = net.get_all_variables()

    def _variables(self, variables):
        values = [variables] if isinstance(variables, str) else list(variables)
        if len(values) != len(set(values)):
            raise ValueError("Variables must not be repeated")
        unknown = set(values) - set(self.variables)
        if unknown:
            raise ValueError(f"Unknown variables: {sorted(unknown)}")
        return sorted(values)

    def _evidence(self, evidence):
        evidence = {} if evidence is None else dict(evidence)
        self._variables(evidence)
        if any(type(v) is not bool for v in evidence.values()):
            raise ValueError("Evidence values must be True or False")
        return evidence

    def d_separated(self, X, Z, Y):
        """Return whether X and Y are independent given Z (original argument order)."""
        x, z, y = (set(self._variables(v)) for v in (X, Z, Y))
        if x & y or x & z or y & z:
            raise ValueError("X, Y and Z must be disjoint")
        if not x or not y:
            return True
        ancestors = x | y | z
        for node in list(ancestors):
            ancestors |= nx.ancestors(self.G, node)
        graph = self.G.subgraph(ancestors)
        moral = graph.to_undirected()
        for node in graph:
            moral.add_edges_from(combinations(graph.predecessors(node), 2))
        moral.remove_nodes_from(z)
        return not any(nx.has_path(moral, a, b) for a in x for b in y)

    @staticmethod
    def _order(factors, hidden, heuristic):
        if not isinstance(heuristic, str):
            order = list(heuristic)
            if len(order) != len(hidden) or set(order) != set(hidden):
                raise ValueError("An explicit order must contain each hidden variable once")
            return order
        if heuristic not in ("min-degree", "min-fill"):
            raise ValueError("Order must be 'min-degree', 'min-fill', or a variable list")
        graph = nx.Graph()
        for factor in factors:
            scope = [c for c in factor if c != "p"]
            graph.add_nodes_from(scope)
            graph.add_edges_from(combinations(scope, 2))
        remaining = set(hidden)
        order = []
        while remaining:
            def score(node):
                neighbours = list(graph.neighbors(node))
                cost = len(neighbours) if heuristic == "min-degree" else sum(
                    not graph.has_edge(a, b) for a, b in combinations(neighbours, 2)
                )
                return cost, node  # Stable tie-breaking makes runs repeatable.

            node = min(remaining, key=score)
            graph.add_edges_from(combinations(list(graph.neighbors(node)), 2))
            graph.remove_node(node)
            remaining.remove(node)
            order.append(node)
        return order

    def _reduced_factors(self, evidence):
        factors = []
        for cpt in self.bn.get_all_cpts().values():
            factor = cpt.copy(deep=True)
            factor['p'] = factor['p'].astype(float)
            for variable, value in evidence.items():
                if variable in factor:
                    factor = factor.loc[factor[variable] == value].drop(columns=variable)
            factors.append(factor.reset_index(drop=True))
        return factors

    def elimination_order(self, query, evidence=None, heuristic="min-fill"):
        query = self._variables(query)
        evidence = self._evidence(evidence)
        if set(query) & set(evidence):
            raise ValueError("Query and evidence must be disjoint")
        hidden = set(self.variables) - set(query) - set(evidence)
        return self._order(self._reduced_factors(evidence), hidden, heuristic)

    def _joint(self, query, evidence, order):
        factors = self._reduced_factors(evidence)
        hidden = set(self.variables) - set(query) - set(evidence)
        for variable in self._order(factors, hidden, order):
            relevant = [f for f in factors if variable in f]
            factors = [f for f in factors if variable not in f]
            factors.append(sum_out(multiply_factors(relevant), variable))
        result = multiply_factors(factors)
        if query:
            result = result.sort_values(query).reset_index(drop=True)
        return result[query + ["p"]]

    def query(self, variables, evidence=None, order="min-fill"):
        """Return P(variables | evidence) as a normalised DataFrame.

        No graph or CPT is changed. Zero-probability evidence raises ValueError.
        An empty query returns the scalar probability 1 if evidence is possible.
        """
        query = self._variables(variables)
        evidence = self._evidence(evidence)
        if set(query) & set(evidence):
            raise ValueError("Query and evidence must be disjoint")
        result = self._joint(query, evidence, order)
        normalizer = float(result["p"].sum())
        if normalizer <= 0:
            raise ValueError("Evidence has zero probability (or numerical underflow)")
        result["p"] /= normalizer
        return result

    def evidence_probability(self, evidence=None, order="min-fill"):
        """Return P(evidence), including zero for impossible evidence."""
        evidence = self._evidence(evidence)
        return float(self._joint([], evidence, order)["p"].sum())

    def marginal_distribution(self, Q, E=None, order="min-fill"):
        """Coursework adapter: Q maps query names to None or a Boolean value.

        Fixed query assignments filter the posterior without renormalising it.
        For example, {'Rain?': True} returns P(Rain?=True | E).
        """
        if not isinstance(Q, dict):
            raise TypeError("Q must map variable names to None, True or False")
        if any(v is not None and type(v) is not bool for v in Q.values()):
            raise ValueError("Query assignments must be None, True or False")
        result = self.query(Q, E, order)
        for variable, value in Q.items():
            if value is not None:
                result = result.loc[result[variable] == value]
        return result.reset_index(drop=True)

    def map(self, variables, evidence=None, order="min-fill"):
        """Find the most likely joint query assignment after summing hidden variables.

        This materialises the query distribution: use only for small query sets.
        Ties choose the first assignment in sorted variable/False-before-True order.
        """
        result = self.query(variables, evidence, order)
        row = result.loc[result["p"].idxmax()]
        return Explanation(
            {v: bool(row[v]) for v in result.columns if v != "p"}, float(row["p"])
        )

    def mpe(self, evidence=None):
        """Find the most likely assignment of all unobserved variables.

        This transparent reference implementation materialises the posterior;
        memory grows exponentially with the number of unobserved variables.
        """
        evidence = self._evidence(evidence)
        return self.map(sorted(set(self.variables) - set(evidence)), evidence)
