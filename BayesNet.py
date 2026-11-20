"""Network container adapted from the KR21 course-provided BayesNet class."""

from typing import List, Tuple, Dict
import networkx as nx
import math
import itertools
import xml.etree.ElementTree as ET
import pandas as pd
from copy import deepcopy


class BayesNet:

    def __init__(self) -> None:
        # initialize graph structure
        self.structure = nx.DiGraph()

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        variables = list(variables)
        edges = list(edges)
        if len(variables) != len(set(variables)) or set(variables) != set(cpts):
            raise ValueError('Variables and CPT keys must match without duplicates')
        if any(a not in cpts or b not in cpts for a, b in edges):
            raise ValueError('Edges must reference declared variables')
        graph = nx.DiGraph()
        graph.add_nodes_from((v, {'cpt': cpts[v].copy(deep=True)}) for v in variables)
        graph.add_edges_from(edges)
        previous = self.structure
        self.structure = graph
        try:
            self.validate()
        except (ValueError, TypeError, KeyError):
            self.structure = previous
            raise

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        try:
            root = ET.parse(file_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f'Invalid XMLBIF document: {exc}') from exc
        networks = root.findall('NETWORK')
        if len(networks) != 1:
            raise ValueError('Expected exactly one NETWORK in the XMLBIF file')
        network = networks[0]
        outcomes = {}
        for variable in network.findall('VARIABLE'):
            name = (variable.findtext('NAME') or '').strip()
            if not name or name in outcomes or name == 'p':
                raise ValueError('Variable names must be unique, nonempty, and not p')
            if variable.get('TYPE', 'nature') != 'nature':
                raise ValueError('Only chance variables (TYPE=nature) are supported')
            labels = [(v.text or '').strip().lower() for v in variable.findall('OUTCOME')]
            if sorted(labels) != ['false', 'true']:
                raise ValueError(f'{name}: only Boolean true/false outcomes are supported')
            outcomes[name] = [label == 'true' for label in labels]

        cpts, edges = {}, []
        for definition in network.findall('DEFINITION'):
            child = (definition.findtext('FOR') or '').strip()
            parents = [(v.text or '').strip() for v in definition.findall('GIVEN')]
            if child not in outcomes or child in cpts:
                raise ValueError('Each declared variable must have exactly one definition')
            if len(parents) != len(set(parents)) or any(p not in outcomes for p in parents):
                raise ValueError(f'{child}: invalid or duplicate parents')
            tables = definition.findall('TABLE')
            if not tables:
                raise ValueError(f'{child}: missing probability table')
            values = [float(v) for v in (tables[-1].text or '').split()]
            columns = parents + [child]
            # XMLBIF counts in declared GIVEN/OUTCOME order, with the child varying fastest.
            worlds = list(itertools.product(*(outcomes[v] for v in columns)))
            if len(worlds) != len(values):
                raise ValueError(f'{child}: incorrect probability table length')
            cpts[child] = pd.DataFrame(
                [list(world) + [p] for world, p in zip(worlds, values)],
                columns=columns + ['p'],
            )
            edges.extend((parent, child) for parent in parents)
        self.create_bn(list(outcomes), edges, cpts)

    def validate(self) -> None:
        """Check the Boolean DAG and complete, normalised conditional tables."""
        if not self.structure or not nx.is_directed_acyclic_graph(self.structure):
            raise ValueError('Expected a nonempty directed acyclic graph')
        for variable in self.structure:
            if not isinstance(variable, str) or not variable or variable == 'p':
                raise ValueError('Variable names must be nonempty strings other than p')
            cpt = self.structure.nodes[variable].get('cpt')
            parents = list(self.structure.predecessors(variable))
            scope = parents + [variable]
            if not isinstance(cpt, pd.DataFrame) or not cpt.columns.is_unique:
                raise ValueError(f'{variable}: expected a table with unique columns')
            if set(cpt.columns) != set(scope + ['p']):
                raise ValueError(f'{variable}: CPT columns must match the variable and parents')
            if len(cpt) != 2 ** len(scope) or cpt.duplicated(scope).any():
                raise ValueError(f'{variable}: CPT must contain every assignment exactly once')
            if any(not pd.api.types.is_bool_dtype(cpt[v]) or cpt[v].isna().any() for v in scope):
                raise ValueError(f'{variable}: CPT assignments must be Boolean')
            try:
                probabilities = cpt['p'].astype(float)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{variable}: invalid probabilities') from exc
            if not probabilities.map(math.isfinite).all() or not probabilities.between(0, 1).all():
                raise ValueError(f'{variable}: probabilities must be finite and between 0 and 1')
            totals = (cpt.assign(p=probabilities).groupby(parents)['p'].sum()
                      if parents else [probabilities.sum()])
            if any(not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9) for total in totals):
                raise ValueError(f'{variable}: conditional probabilities must sum to one')

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.successors(variable)]

    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all cps in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = [column for column in self.get_cpt(var).columns if column != 'p']
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph

    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    def draw_structure(self) -> None:
        """
        Visualize structure of the BN.
        """
        import matplotlib.pyplot as plt

        nx.draw(self.structure, with_labels=True, node_size=3000)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])

