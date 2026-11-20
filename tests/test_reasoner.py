"""Check inference against independent exhaustive enumeration, not its own helpers."""

from itertools import product
from pathlib import Path
import random
import subprocess
import sys

import pandas as pd
import pytest

from BayesNet import BayesNet
from BNReasoner import BNReasoner, multiply_factors, sum_out
from MAP import MAP
from MPE import MPE

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = sorted((ROOT / 'testing').glob('*.BIFXML'))


def enumerate_posterior(bn, query, evidence):
    """Multiply one local probability per node in every full Boolean world."""
    variables = bn.get_all_variables()
    query = sorted(query)
    totals = {values: 0.0 for values in product([False, True], repeat=len(query))}
    for values in product([False, True], repeat=len(variables)):
        world = dict(zip(variables, values))
        if any(world[v] != value for v, value in evidence.items()):
            continue
        probability = 1.0
        for cpt in bn.get_all_cpts().values():
            for _, row in cpt.iterrows():
                if all(row[v] == world[v] for v in cpt if v != 'p'):
                    probability *= float(row['p'])
                    break
            else:
                raise AssertionError('Incomplete CPT in test input')
        totals[tuple(world[v] for v in query)] += probability
    normalizer = sum(totals.values())
    return {k: v / normalizer for k, v in totals.items()}, normalizer


def table_dict(table):
    return {tuple(row[:-1]): float(row[-1]) for row in table.itertuples(index=False, name=None)}


def make_network(edges, nodes, seed=1):
    rng = random.Random(seed)
    cpts = {}
    for node in nodes:
        parents = [a for a, b in edges if b == node]
        rows = []
        for values in product([False, True], repeat=len(parents)):
            p = rng.uniform(0.05, 0.95)
            rows.extend([list(values) + [False, 1-p], list(values) + [True, p]])
        cpts[node] = pd.DataFrame(rows, columns=parents + [node, 'p'])
    bn = BayesNet()
    bn.create_bn(nodes, edges, cpts)
    return bn


@pytest.mark.parametrize('path', NETWORKS, ids=lambda p: p.stem)
@pytest.mark.parametrize('order', ['min-fill', 'min-degree'])
def test_supplied_networks_against_enumeration(path, order):
    r = BNReasoner(path)
    for variable in r.variables:
        others = [v for v in r.variables if v != variable]
        for evidence in ({}, {others[0]: True}, {others[0]: False, others[-1]: True}):
            expected, mass = enumerate_posterior(r.bn, [variable], evidence)
            assert table_dict(r.query([variable], evidence, order)) == pytest.approx(expected)
            assert r.evidence_probability(evidence, order) == pytest.approx(mass)


@pytest.mark.parametrize('seed', range(8))
def test_generated_networks_and_explicit_order(seed):
    rng = random.Random(seed)
    nodes = list('ABCDE')
    edges = [(a, b) for i, a in enumerate(nodes) for b in nodes[i+1:] if rng.random() < 0.4]
    r = BNReasoner(make_network(edges, nodes, seed))
    query, evidence = ['B', 'E'], {'A': bool(seed % 2)}
    expected, _ = enumerate_posterior(r.bn, query, evidence)
    for order in ('min-fill', 'min-degree', ['D', 'C']):
        assert table_dict(r.query(query, evidence, order)) == pytest.approx(expected)


def test_xml_outcome_and_parent_order():
    r = BNReasoner(ROOT / 'testing/dog_problem.BIFXML')
    assert r.query(['family-out']).loc[1, 'p'] == pytest.approx(0.15)
    assert r.query(['bowel-problem']).loc[1, 'p'] == pytest.approx(0.01)
    cpt = r.bn.get_cpt('dog-out')
    expected = {(True, True): 0.99, (True, False): 0.97,
                (False, True): 0.9, (False, False): 0.3}
    for (bowel, family), p in expected.items():
        row = cpt.loc[(cpt['bowel-problem'] == bowel) & (cpt['family-out'] == family) & cpt['dog-out']]
        assert float(row['p'].iloc[0]) == pytest.approx(p)
    # Hand calculation: P(bark, family out) / P(bark).
    bark_given_out = (0.01 * 0.99 + 0.99 * 0.9) * 0.7 + (0.01 * 0.01 + 0.99 * 0.1) * 0.01
    bark_given_in = (0.01 * 0.97 + 0.99 * 0.3) * 0.7 + (0.01 * 0.03 + 0.99 * 0.7) * 0.01
    expected_p = 0.15 * bark_given_out / (0.15 * bark_given_out + 0.85 * bark_given_in)
    assert r.query(['family-out'], {'hear-bark': True}).loc[1, 'p'] == pytest.approx(expected_p)


def test_scalar_disjoint_and_overlapping_factors():
    a = pd.DataFrame({'A': [False, True], 'p': [0.3, 0.7]})
    b = pd.DataFrame({'B': [False, True], 'p': [0.6, 0.4]})
    assert table_dict(multiply_factors([a, b])) == pytest.approx({
        (False, False): 0.18, (False, True): 0.12, (True, False): 0.42, (True, True): 0.28})
    assert list(multiply_factors([a, a])['p']) == pytest.approx([0.09, 0.49])
    assert list(sum_out(a, 'A')['p']) == [1.0]
    assert list(multiply_factors([])['p']) == [1.0]
    assert list(multiply_factors([a, pd.DataFrame({'p': [0.5]})])['p']) == pytest.approx([0.15, 0.35])


@pytest.mark.parametrize('edges,nodes,x,z,y,expected', [
    ([('A','B'),('B','C')], list('ABC'), {'A'}, set(), {'C'}, False),
    ([('A','B'),('B','C')], list('ABC'), {'A'}, {'B'}, {'C'}, True),
    ([('B','A'),('B','C')], list('ABC'), {'A'}, {'B'}, {'C'}, True),
    ([('A','B'),('C','B')], list('ABC'), {'A'}, set(), {'C'}, True),
    ([('A','B'),('C','B')], list('ABC'), {'A'}, {'B'}, {'C'}, False),
    ([('A','B'),('C','B'),('B','D'),('D','E')], list('ABCDE'), {'A'}, {'E'}, {'C'}, False),
    ([], list('AB'), {'A'}, set(), {'B'}, True),
    ([], list('AB'), set(), set(), {'B'}, True),
])
def test_d_separation(edges, nodes, x, z, y, expected):
    r = BNReasoner(make_network(edges, nodes))
    assert r.d_separated(x, z, y) is expected


def test_map_sums_hidden_variables_before_maximising():
    bn = BayesNet()
    bn.create_bn(['A', 'B'], [('A', 'B')], {
        'A': pd.DataFrame({'A': [False, True], 'p': [0.6, 0.4]}),
        'B': pd.DataFrame({'A': [False, False, True, True], 'B': [False, True, False, True],
                           'p': [0.5, 0.5, 0.01, 0.99]}),
    })
    assert MAP().run(bn, ['A']).assignment == {'A': False}
    assert MAP().run(bn, ['A']).probability == pytest.approx(0.6)
    assert MPE().run(bn).assignment == {'A': True, 'B': True}
    assert MPE().run(bn).probability == pytest.approx(0.396)


def test_conditional_explanation_and_no_mutation():
    r = BNReasoner(NETWORKS[0])
    evidence = {r.variables[0]: True}
    before = {k: v.copy(deep=True) for k, v in r.bn.get_all_cpts().items()}
    edges = list(r.G.edges)
    expected, _ = enumerate_posterior(r.bn, set(r.variables) - set(evidence), evidence)
    result = r.mpe(evidence)
    assert result.probability == pytest.approx(max(expected.values()))
    assert expected[tuple(result.assignment.values())] == pytest.approx(result.probability)
    r.query([], evidence)
    r.d_separated({r.variables[1]}, set(), {r.variables[2]})
    for key, cpt in before.items():
        pd.testing.assert_frame_equal(cpt, r.bn.get_cpt(key))
    assert edges == list(r.G.edges)


def test_impossible_evidence_empty_query_and_all_observed():
    bn = BayesNet()
    bn.create_bn(['A'], [], {'A': pd.DataFrame({'A': [False, True], 'p': [1.0, 0.0]})})
    r = BNReasoner(bn)
    assert r.evidence_probability({'A': True}) == 0
    with pytest.raises(ValueError, match='zero probability'):
        r.query([], {'A': True})
    assert list(r.query([], {'A': False})['p']) == [1.0]
    assert r.mpe({'A': False}).assignment == {}
    assert r.mpe({'A': False}).probability == 1.0


@pytest.mark.parametrize('query,evidence,order', [
    (['missing'], {}, 'min-fill'), (['A','A'], {}, 'min-fill'),
    (['A'], {'A': True}, 'min-fill'), (['A'], {'B': 1}, 'min-fill'),
    (['A'], {'B': None}, 'min-fill'), (['A'], {}, 'unknown'),
    (['A'], {}, []), (['A'], {}, ['B','B']),
])
def test_reject_bad_queries(query, evidence, order):
    r = BNReasoner(make_network([], list('AB')))
    with pytest.raises(ValueError):
        r.query(query, evidence, order)


def test_original_marginal_interface():
    r = BNReasoner(ROOT / 'testing/dog_problem.BIFXML')
    result = r.marginal_distribution({'family-out': True}, {})
    assert float(result['p'].iloc[0]) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        r.d_separated({'family-out'}, {'family-out'}, {'dog-out'})


def test_order_updates_degrees_and_is_repeatable():
    factors = [pd.DataFrame(columns=['A','B','p']), pd.DataFrame(columns=['B','C','p']),
               pd.DataFrame(columns=['C','D','p'])]
    assert BNReasoner._order(factors, set('ABCD'), 'min-degree') == list('ABCD')
    r = BNReasoner(ROOT / 'testing/lecture_example.BIFXML')
    for heuristic in ('min-degree', 'min-fill'):
        assert r.elimination_order(['Rain?'], heuristic=heuristic) == r.elimination_order(['Rain?'], heuristic=heuristic)


def test_min_fill_and_min_degree_can_choose_different_nodes():
    factors = [pd.DataFrame(columns=['A','B','C','p']), pd.DataFrame(columns=['D','Z','p'])]
    assert BNReasoner._order(factors, set('ABCDZ'), 'min-fill')[0] == 'A'
    assert BNReasoner._order(factors, set('ABCDZ'), 'min-degree')[0] == 'D'


def test_cli_works_from_another_directory(tmp_path):
    run = subprocess.run([sys.executable, str(ROOT / 'demo.py')], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert '0.150000' in run.stdout and '0.334636' in run.stdout
    bad = subprocess.run([sys.executable, str(ROOT / 'demo.py'), '--query', 'family-out',
                          '--evidence', 'hear-bark=true', 'hear-bark=false'], capture_output=True, text=True)
    assert bad.returncode == 2 and 'only once' in bad.stderr


def test_imports_have_no_demo_side_effects(tmp_path):
    run = subprocess.run([sys.executable, '-c',
        f'import sys; sys.path.insert(0, {str(ROOT)!r}); import BNReasoner, BayesNet, MAP, MPE, demo'],
        cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout == ''
