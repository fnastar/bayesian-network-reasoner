from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from BayesNet import BayesNet
from BNReasoner import BNReasoner


def xml(labels=('false', 'true'), table='0.8 0.2', extra=''):
    return f'''<BIF VERSION="0.3"><NETWORK><NAME>Test</NAME>
      <VARIABLE TYPE="nature"><NAME>A</NAME>
        <OUTCOME>{labels[0]}</OUTCOME><OUTCOME>{labels[1]}</OUTCOME></VARIABLE>
      <DEFINITION><FOR>A</FOR><TABLE>{table}</TABLE></DEFINITION>{extra}
    </NETWORK></BIF>'''


def test_reversed_outcome_declaration_and_reloading():
    bn = BayesNet()
    bn.load_from_bifxml(StringIO(xml()))
    assert BNReasoner(bn).query(['A']).loc[1, 'p'] == pytest.approx(0.2)
    bn.load_from_bifxml(StringIO(xml(labels=('true', 'false'))))
    assert BNReasoner(bn).query(['A']).loc[1, 'p'] == pytest.approx(0.8)


@pytest.mark.parametrize('content', [
    xml(labels=('low', 'high')), xml(table='0.2'), xml(table='0.8 0.8'),
    xml(table='nan 0.2'), xml(table='-0.2 1.2'),
    xml(extra='<DEFINITION><FOR>A</FOR><TABLE>0.1 0.9</TABLE></DEFINITION>'),
    xml(extra='<VARIABLE><NAME>B</NAME><OUTCOME>true</OUTCOME><OUTCOME>false</OUTCOME></VARIABLE>'),
    xml().replace('<FOR>A</FOR>', '<FOR>A</FOR><GIVEN>missing</GIVEN>'),
    xml().replace('TYPE="nature"', 'TYPE="decision"'),
    '<BIF/>', '<BIF>',
])
def test_invalid_xml_networks_are_rejected(content):
    with pytest.raises(ValueError):
        BayesNet().load_from_bifxml(StringIO(content))


def test_last_xml_table_is_used():
    content = xml().replace('</DEFINITION>', '<TABLE>0.3 0.7</TABLE></DEFINITION>')
    bn = BayesNet()
    bn.load_from_bifxml(StringIO(content))
    assert BNReasoner(bn).query(['A']).loc[1, 'p'] == pytest.approx(0.7)


def test_programmatic_probabilities_are_used_numerically():
    bn = BayesNet()
    bn.create_bn(['A'], [], {'A': pd.DataFrame({'p': ['0.8', '0.2'], 'A': [False, True]})})
    assert BNReasoner(bn).query(['A']).loc[1, 'p'] == pytest.approx(0.2)
    assert set(bn.get_interaction_graph()) == {'A'}


def test_reject_cycle_and_invalid_cpt_without_replacing_network():
    bn = BayesNet()
    bn.load_from_bifxml(StringIO(xml()))
    previous = bn.structure
    with pytest.raises(ValueError):
        bn.create_bn(['A'], [('A', 'A')], {'A': pd.DataFrame({'A': [False, True], 'p': [0.5, 0.5]})})
    assert bn.structure is previous
    with pytest.raises(ValueError):
        bn.create_bn(['A'], [], {'A': pd.DataFrame({'A': [False, False], 'p': [0.5, 0.5]})})
    assert bn.structure is previous
    with pytest.raises(ValueError):
        bn.create_bn(['A'], [('A', 'B')], {'A': pd.DataFrame({'A': [False, True], 'p': [0.5, 0.5]})})


def test_archived_hiring_draft_is_explicitly_rejected():
    path = Path(__file__).resolve().parents[1] / 'archive/hiring-draft.BIFXML'
    with pytest.raises(ValueError, match='definition'):
        BNReasoner(path)
