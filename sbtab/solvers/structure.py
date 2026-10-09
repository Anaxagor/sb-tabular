from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class LearnedDAG:
    """
    A DAG over columns, stored as explicit ORDERED parent lists.

    Field models are fitted on features [x, parents...]; the parent order is part
    of the model contract. It is stored verbatim and never re-derived from a
    rebuilt graph, whose predecessor order is not guaranteed to match.
    """
    order: List[str]
    parents: Dict[str, List[str]]
    fit_row_count: int = 0
    n_bins: int = 0
    edges: List[tuple] = field(default_factory=list)

    def graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(self.order)
        G.add_edges_from((p, c) for c, ps in self.parents.items() for p in ps)
        return G

    def validate(self) -> None:
        G = self.graph()
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("learned structure is not acyclic")
        if set(self.order) != set(self.parents):
            raise ValueError("generation order and parent map cover different columns")
        seen = set()
        for col in self.order:
            missing = [p for p in self.parents[col] if p not in seen]
            if missing:
                raise ValueError(f"column {col!r} is generated before its parents {missing}")
            seen.add(col)

    def state(self) -> dict:
        return {"order": list(self.order), "parents": {k: list(v) for k, v in self.parents.items()},
                "fit_row_count": int(self.fit_row_count), "n_bins": int(self.n_bins),
                "edges": [tuple(e) for e in self.edges]}

    @classmethod
    def from_state(cls, state: dict) -> "LearnedDAG":
        dag = cls(order=list(state["order"]), parents={k: list(v) for k, v in state["parents"].items()},
                  fit_row_count=int(state.get("fit_row_count", 0)), n_bins=int(state.get("n_bins", 0)),
                  edges=[tuple(e) for e in state.get("edges", [])])
        dag.validate()
        return dag


def learn_dag(df: pd.DataFrame, n_bins: int = 5) -> LearnedDAG:
    """
    Hill-climb/BIC structure search on a quantile-binned copy of ``df``.

    ``df`` must contain ONLY the current training rows: the discretiser and the
    structure search are both fitted on exactly the rows passed in.
    """
    from pgmpy.estimators import HillClimbSearch
    try:
        from pgmpy.estimators import BicScore
    except ImportError:
        from pgmpy.estimators import BIC as BicScore
    from sklearn.preprocessing import KBinsDiscretizer

    cols = list(df.columns)
    # Constant columns cannot be quantile-binned and carry no dependence.
    varying = [c for c in cols if df[c].nunique(dropna=False) > 1]
    binned = pd.DataFrame(index=df.index)
    if varying:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned = pd.DataFrame(disc.fit_transform(df[varying].to_numpy(dtype=float)), columns=varying).astype(int)

    edges = []
    if len(varying) >= 2:
        hc = HillClimbSearch(binned)
        best = hc.estimate(scoring_method=BicScore(binned), show_progress=False)
        # Keep the dataframe's labels: coercing only edge endpoints to strings
        # creates extra graph nodes for integer-labeled columns.
        edges = list(best.edges())

    G = nx.DiGraph()
    G.add_nodes_from(cols)
    G.add_edges_from((a, b) for a, b in edges)
    # HillClimbSearch returns a DAG; this is a safeguard, not the normal path.
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(cycle[-1][0], cycle[-1][1])

    order = list(nx.lexicographical_topological_sort(G, key=lambda c: cols.index(c)))
    parents = {c: sorted(G.predecessors(c), key=lambda p: cols.index(p)) for c in cols}
    dag = LearnedDAG(order=order, parents=parents, fit_row_count=len(df), n_bins=n_bins, edges=list(G.edges()))
    dag.validate()
    return dag


def parent_matrix(frame: pd.DataFrame, parents: List[str], n: int) -> np.ndarray:
    if not parents:
        return np.empty((n, 0), dtype=np.float32)
    return frame[parents].to_numpy(dtype=np.float32)
