"""Dharma-Topology Protocol — three-pillar evaluation on arbitrary graphs.

Maps code/system quality to the KCL circuit ODE:
  C * dV/dt = -V/R_leak + g'(V) * G_prec * error

Three Pillars (Sila):
  1. Karma Isolation (因果の局所化): side-effect quarantine score
  2. G_prec Topology (縁起の中道): dependency graph block-diagonality
  3. Delete-ability (無常の受容): module removal impact

Output: DharmaTelemetry JSON with three pillar scores and ODE convergence data.
"""

from __future__ import annotations

import collections
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

import numpy as np
from scipy import sparse

from lmm.dharma.energy import (
    DharmaEnergyTerm,
    MathProperty,
    _csr_scale_data,
    _resize_sparse,
)
from lmm.dharma.fep import solve_fep_kcl_analog


def _remove_diagonal(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """Remove diagonal entries from a sparse matrix (shim-compatible)."""
    coo = m.tocoo()
    mask = [int(r) != int(c) for r, c in zip(coo.row, coo.col)]
    rows = [int(r) for r, keep in zip(coo.row, mask) if keep]
    cols = [int(c) for c, keep in zip(coo.col, mask) if keep]
    vals = [float(v) for v, keep in zip(coo.data, mask) if keep]
    if not vals:
        return sparse.csr_matrix(([], ([], [])), shape=m.shape)
    return sparse.csr_matrix((vals, (rows, cols)), shape=m.shape)


def _symmetrize(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """Symmetrize and remove diagonal (shim-compatible)."""
    sym = (adj + adj.T) * 0.5
    return _remove_diagonal(sym)


# ===================================================================
# Telemetry dataclass
# ===================================================================


@dataclass
class DharmaTelemetry:
    """Dharma topology evaluation telemetry.

    Attributes
    ----------
    karma_isolation : float
        [0, 1] how well side effects are quarantined.
    gprec_topology : float
        [0, 1] dependency graph sparsity / block-diagonality.
    deleteability : float
        [0, 1] average ease of removing any single node.
    overall_dharma : float
        [0, 1] geometric mean of the three pillars.
    ode_steps : int
        Number of KCL ODE steps to convergence.
    ode_final_power : float
        Final dissipation power (lower = more converged).
    V_final : list[float]
        Per-node membrane potentials after ODE convergence.
    pillar_details : dict | None
        Per-node breakdown if requested.
    """

    karma_isolation: float
    gprec_topology: float
    deleteability: float
    overall_dharma: float
    ode_steps: int
    ode_final_power: float
    V_final: list[float]
    pillar_details: dict | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict (dharma_telemetry protocol)."""
        d = asdict(self)
        d["protocol"] = "dharma_topology_v1"
        return d


# ===================================================================
# Pure metric functions
# ===================================================================


def _bfs_components(adj: sparse.csr_matrix) -> list[list[int]]:
    """Find connected components via BFS (shim-compatible, no eigsh)."""
    n = adj.shape[0]
    visited = [False] * n
    components: list[list[int]] = []
    # Build symmetric adjacency for undirected component search
    sym = adj + adj.T
    for start in range(n):
        if visited[start]:
            continue
        comp: list[int] = []
        queue = collections.deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            comp.append(node)
            row = sym.getrow(node).tocoo()
            for neighbor in row.col:
                nb = int(neighbor)
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)
    return components


def compute_karma_isolation(
    adj: sparse.csr_matrix,
    boundary_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Karma Isolation: measure side-effect quarantine quality.

    For each boundary node, computes the ratio of internal coupling
    (within its cluster) to total coupling. Non-boundary nodes score 1.0.

    Parameters
    ----------
    adj : (n, n) sparse adjacency/dependency matrix
    boundary_mask : (n,) bool array, True = boundary node. None = auto-detect.

    Returns
    -------
    score : float in [0, 1], higher = better isolation
    per_node : (n,) per-node isolation scores
    """
    n = adj.shape[0]
    if n == 0:
        return 1.0, np.array([])

    if adj.nnz == 0:
        return 1.0, np.ones(n)

    # Compute degrees (sum of absolute values for weighted graphs)
    out_degree = np.asarray(np.abs(adj).sum(axis=1)).flatten()

    if boundary_mask is None:
        # Auto-detect: nodes with above-median out-degree are boundary
        sorted_deg = sorted(float(d) for d in out_degree)
        mid = len(sorted_deg) // 2
        if len(sorted_deg) % 2 == 1:
            median_deg = sorted_deg[mid]
        else:
            median_deg = (sorted_deg[mid - 1] + sorted_deg[mid]) / 2.0
        boundary_mask = out_degree > median_deg

    # Find connected components for cluster assignment
    components = _bfs_components(adj)
    node_cluster = np.zeros(n, dtype=int)
    for cid, comp in enumerate(components):
        for node in comp:
            node_cluster[node] = cid

    per_node = np.ones(n)

    for i in range(n):
        if not boundary_mask[i]:
            continue
        row = adj.getrow(i).tocoo()
        if len(row.data) == 0:
            continue
        total_weight = sum(abs(float(v)) for v in row.data)
        if total_weight < 1e-15:
            continue
        internal_weight = 0.0
        my_cluster = node_cluster[i]
        for c, v in zip(row.col, row.data):
            if node_cluster[int(c)] == my_cluster:
                internal_weight += abs(float(v))
        per_node[i] = internal_weight / total_weight

    return float(per_node.mean()), per_node


def compute_gprec_topology(
    adj: sparse.csr_matrix,
) -> tuple[float, dict]:
    """G_prec Topology: measure dependency graph block-diagonality.

    Computes sparsity, connected component structure, and block-diagonal
    ratio to assess modularity.

    Returns
    -------
    score : float in [0, 1], higher = more modular/sparse
    details : dict with density, n_components, block_diagonal_ratio
    """
    n = adj.shape[0]
    if n == 0:
        return 1.0, {"density": 0.0, "n_components": 0, "block_diagonal_ratio": 1.0}

    if adj.nnz == 0:
        return 1.0, {"density": 0.0, "n_components": n, "block_diagonal_ratio": 1.0}

    # 1. Density: off-diagonal density
    max_edges = n * n - n
    density = adj.nnz / max(max_edges, 1)
    sparsity_score = 1.0 - min(density, 1.0)

    # 2. Connected components
    components = _bfs_components(adj)
    n_components = len(components)
    # Normalize: more components relative to n is better
    component_score = min((n_components - 1) / max(n - 1, 1), 1.0)

    # 3. Block-diagonal ratio: fraction of edges within components
    node_cluster = np.zeros(n, dtype=int)
    for cid, comp in enumerate(components):
        for node in comp:
            node_cluster[cid] = cid  # intentionally using comp for cluster
    # Re-assign correctly
    for cid, comp in enumerate(components):
        for node in comp:
            node_cluster[node] = cid

    coo = adj.tocoo()
    total_weight = 0.0
    internal_weight = 0.0
    for r, c, v in zip(coo.row, coo.col, coo.data):
        w = abs(float(v))
        total_weight += w
        if node_cluster[int(r)] == node_cluster[int(c)]:
            internal_weight += w

    block_diag_ratio = internal_weight / max(total_weight, 1e-15)

    score = 0.4 * sparsity_score + 0.3 * component_score + 0.3 * block_diag_ratio

    return float(np.clip(score, 0.0, 1.0)), {
        "density": density,
        "n_components": n_components,
        "block_diagonal_ratio": block_diag_ratio,
    }


def compute_deleteability(
    adj: sparse.csr_matrix,
    *,
    method: str = "degree",
    G_prec: float = 5.0,
    tau_leak: float = 1.0,
    max_steps: int = 50,
) -> tuple[float, np.ndarray]:
    """Delete-ability: measure module removal impact.

    Parameters
    ----------
    adj : (n, n) sparse adjacency/dependency matrix
    method : "degree" for fast heuristic, "fep" for ODE-based analysis
    G_prec : precision conductance (fep method only)
    tau_leak : leak time constant (fep method only)
    max_steps : ODE steps per node (fep method only)

    Returns
    -------
    score : float in [0, 1], higher = better delete-ability
    per_node : (n,) per-node delete-ability scores
    """
    n = adj.shape[0]
    if n == 0:
        return 1.0, np.array([])

    if adj.nnz == 0:
        return 1.0, np.ones(n)

    if method == "fep":
        return _deleteability_fep(adj, G_prec=G_prec, tau_leak=tau_leak, max_steps=max_steps)

    return _deleteability_degree(adj)


def _deleteability_degree(adj: sparse.csr_matrix) -> tuple[float, np.ndarray]:
    """Fast degree-based delete-ability heuristic."""
    n = adj.shape[0]
    abs_adj = abs(adj)
    out_deg = np.asarray(abs_adj.sum(axis=1)).flatten()
    in_deg = np.asarray(abs_adj.sum(axis=0)).flatten()
    total_deg = out_deg + in_deg
    max_deg = float(total_deg.max())
    if max_deg < 1e-15:
        return 1.0, np.ones(n)
    impact = total_deg / (2.0 * max_deg)
    per_node = 1.0 - impact
    return float(per_node.mean()), per_node


def _deleteability_fep(
    adj: sparse.csr_matrix,
    *,
    G_prec: float = 5.0,
    tau_leak: float = 1.0,
    max_steps: int = 50,
) -> tuple[float, np.ndarray]:
    """FEP perturbation-based delete-ability.

    For each node i, inject a unit perturbation and measure system response.
    """
    n = adj.shape[0]
    # Symmetrize for ODE (shim-compatible)
    sym = _symmetrize(adj)

    per_node = np.zeros(n)

    for i in range(n):
        V_s = np.zeros(n)
        V_s[i] = 1.0  # unit perturbation at node i
        V_mu, _, _, power = solve_fep_kcl_analog(
            V_s=V_s,
            J_dynamic=sym,
            n=n,
            G_prec_base=G_prec,
            tau_leak=tau_leak,
            dt=0.01,
            max_steps=max_steps,
            nirvana_threshold=1e-4,
        )
        # Impact = how much the rest of the system responds
        response = float(np.sqrt(np.dot(V_mu, V_mu)))
        # Normalize: higher response = harder to delete
        per_node[i] = response

    # Normalize to [0, 1] and invert
    max_resp = float(per_node.max())
    if max_resp > 1e-15:
        per_node = 1.0 - per_node / max_resp
    else:
        per_node = np.ones(n)

    return float(per_node.mean()), per_node


# ===================================================================
# TopologyEvaluator
# ===================================================================


class TopologyEvaluator:
    """Evaluate a node-edge graph through the Dharma-Topology three-pillar lens.

    Parameters
    ----------
    G_prec : float
        Precision conductance for KCL ODE.
    tau_leak : float
        Leak time constant.
    dt : float
        ODE integration time step.
    max_steps : int
        Maximum ODE steps.
    nirvana_threshold : float
        Convergence threshold for ODE stopping.
    deleteability_method : str
        "degree" for fast heuristic, "fep" for ODE-based analysis.
    """

    def __init__(
        self,
        *,
        G_prec: float = 5.0,
        tau_leak: float = 1.0,
        dt: float = 0.01,
        max_steps: int = 200,
        nirvana_threshold: float = 1e-4,
        deleteability_method: str = "degree",
    ):
        self.G_prec = G_prec
        self.tau_leak = tau_leak
        self.dt = dt
        self.max_steps = max_steps
        self.nirvana_threshold = nirvana_threshold
        self.deleteability_method = deleteability_method

    def evaluate(
        self,
        adj: sparse.csr_matrix,
        *,
        boundary_mask: np.ndarray | None = None,
        node_labels: Sequence[str] | None = None,
        include_details: bool = False,
    ) -> DharmaTelemetry:
        """Run full three-pillar evaluation on an adjacency graph.

        Parameters
        ----------
        adj : (n, n) sparse adjacency matrix (dependency graph)
        boundary_mask : optional (n,) bool, True = side-effect boundary node
        node_labels : optional node names for detailed output
        include_details : if True, populate pillar_details with per-node breakdown

        Returns
        -------
        DharmaTelemetry with all three pillar scores and ODE convergence data.
        """
        n = adj.shape[0]

        # 1. Three-pillar metrics
        karma_score, karma_per_node = compute_karma_isolation(adj, boundary_mask)
        gprec_score, gprec_details = compute_gprec_topology(adj)
        del_score, del_per_node = compute_deleteability(
            adj,
            method=self.deleteability_method,
            G_prec=self.G_prec,
            tau_leak=self.tau_leak,
            max_steps=min(self.max_steps, 50),
        )

        # 2. ODE convergence analysis on the full graph
        ode_steps = 0
        ode_final_power = 0.0
        V_final_list: list[float] = []

        if n > 0:
            # Symmetrize for ODE (shim-compatible)
            sym = _symmetrize(adj)

            # Uniform sensory input: steady-state health check
            V_s = np.ones(n) * 0.5

            V_mu, _, steps, power = solve_fep_kcl_analog(
                V_s=V_s,
                J_dynamic=sym,
                n=n,
                G_prec_base=self.G_prec,
                tau_leak=self.tau_leak,
                dt=self.dt,
                max_steps=self.max_steps,
                nirvana_threshold=self.nirvana_threshold,
            )
            ode_steps = steps
            ode_final_power = power[-1] if power else 0.0
            V_final_list = [float(v) for v in V_mu]
        else:
            V_final_list = []

        # 3. Overall score: geometric mean
        if karma_score > 0 and gprec_score > 0 and del_score > 0:
            overall = (karma_score * gprec_score * del_score) ** (1.0 / 3.0)
        else:
            overall = 0.0

        # 4. Details
        details = None
        if include_details:
            labels = list(node_labels) if node_labels else [str(i) for i in range(n)]
            details = {
                "nodes": {
                    labels[i]: {
                        "karma_isolation": float(karma_per_node[i]),
                        "deleteability": float(del_per_node[i]),
                        "V_final": V_final_list[i] if i < len(V_final_list) else 0.0,
                    }
                    for i in range(n)
                },
                "gprec": gprec_details,
            }

        return DharmaTelemetry(
            karma_isolation=karma_score,
            gprec_topology=gprec_score,
            deleteability=del_score,
            overall_dharma=overall,
            ode_steps=ode_steps,
            ode_final_power=ode_final_power,
            V_final=V_final_list,
            pillar_details=details,
        )


# ===================================================================
# Topology History — time-series tracking with trend analysis
# ===================================================================


class TopologyHistory:
    """Sliding-window history of DharmaTelemetry snapshots.

    Follows the deque pattern from DriftDetector: records timestamped
    snapshots and computes moving-window trends for each pillar.

    Parameters
    ----------
    maxlen : int
        Maximum number of snapshots to retain.
    """

    def __init__(self, maxlen: int = 200):
        self._history: collections.deque[tuple[float, DharmaTelemetry]] = collections.deque(
            maxlen=maxlen
        )

    def record(self, telemetry: DharmaTelemetry) -> None:
        """Record a new telemetry snapshot with current timestamp."""
        self._history.append((time.time(), telemetry))

    def latest(self) -> DharmaTelemetry | None:
        """Return the most recent telemetry, or None if empty."""
        if not self._history:
            return None
        return self._history[-1][1]

    @property
    def count(self) -> int:
        return len(self._history)

    def trend(self, window: int = 10) -> dict:
        """Compute trend over the last *window* snapshots.

        Returns a dict with per-pillar trend slopes and an overall
        direction indicator: "improving", "stable", or "degrading".
        """
        if len(self._history) < 2:
            return {
                "karma_trend": 0.0,
                "gprec_trend": 0.0,
                "deleteability_trend": 0.0,
                "overall_trend": 0.0,
                "direction": "stable",
                "n_snapshots": len(self._history),
            }

        recent = list(self._history)[-window:]
        n = len(recent)

        def _slope(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            x_mean = (len(values) - 1) / 2.0
            y_mean = sum(values) / len(values)
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            den = sum((i - x_mean) ** 2 for i in range(len(values)))
            if abs(den) < 1e-15:
                return 0.0
            return num / den

        karma_vals = [t.karma_isolation for _, t in recent]
        gprec_vals = [t.gprec_topology for _, t in recent]
        del_vals = [t.deleteability for _, t in recent]
        overall_vals = [t.overall_dharma for _, t in recent]

        karma_trend = _slope(karma_vals)
        gprec_trend = _slope(gprec_vals)
        del_trend = _slope(del_vals)
        overall_trend = _slope(overall_vals)

        # Direction based on overall trend magnitude
        if overall_trend > 0.005:
            direction = "improving"
        elif overall_trend < -0.005:
            direction = "degrading"
        else:
            direction = "stable"

        return {
            "karma_trend": karma_trend,
            "gprec_trend": gprec_trend,
            "deleteability_trend": del_trend,
            "overall_trend": overall_trend,
            "direction": direction,
            "n_snapshots": n,
        }

    def to_json(self, last_n: int | None = None) -> dict:
        """Serialize history to JSON-compatible dict.

        Parameters
        ----------
        last_n : int | None
            If given, only include the last *last_n* snapshots.
        """
        items = list(self._history)
        if last_n is not None:
            items = items[-last_n:]

        snapshots = [
            {
                "timestamp": ts,
                "karma_isolation": t.karma_isolation,
                "gprec_topology": t.gprec_topology,
                "deleteability": t.deleteability,
                "overall_dharma": t.overall_dharma,
            }
            for ts, t in items
        ]

        return {
            "snapshots": snapshots,
            "trend": self.trend(),
            "count": len(self._history),
        }

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        """Save history to a JSON file."""
        data = {
            "version": 1,
            "entries": [
                {
                    "timestamp": ts,
                    "karma_isolation": t.karma_isolation,
                    "gprec_topology": t.gprec_topology,
                    "deleteability": t.deleteability,
                    "overall_dharma": t.overall_dharma,
                    "ode_steps": t.ode_steps,
                    "ode_final_power": t.ode_final_power,
                }
                for ts, t in self._history
            ],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(p)

    def load(self, path: str | Path) -> int:
        """Load history from a JSON file.  Returns number of entries loaded."""
        p = Path(path)
        if not p.exists():
            return 0
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            entries = raw.get("entries", [])
            for e in entries:
                t = DharmaTelemetry(
                    karma_isolation=e["karma_isolation"],
                    gprec_topology=e["gprec_topology"],
                    deleteability=e["deleteability"],
                    overall_dharma=e["overall_dharma"],
                    ode_steps=e.get("ode_steps", 0),
                    ode_final_power=e.get("ode_final_power", 0.0),
                    V_final=[],
                )
                self._history.append((e["timestamp"], t))
            return len(entries)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.debug("Failed to load topology history from %s", path, exc_info=True)
            return 0


# ===================================================================
# Topology Drift Detector — rapid-change alerts
# ===================================================================


@dataclass
class TopologyAlert:
    """Alert fired when a topology pillar changes rapidly."""

    timestamp: float
    pillar: str
    previous_value: float
    current_value: float
    delta: float
    severity: str  # "warning" | "critical"
    message: str


class TopologyDriftDetector:
    """Detects sudden changes in topology pillar scores.

    Fires alerts when the delta between consecutive evaluations exceeds
    configurable thresholds.  Follows the DriftDetector z-score pattern
    but operates on structured pillar scores rather than surprise scalars.

    Parameters
    ----------
    warning_threshold : float
        Absolute delta to trigger a warning alert.
    critical_threshold : float
        Absolute delta to trigger a critical alert.
    cooldown : float
        Minimum seconds between alerts for the same pillar.
    """

    PILLARS = ("karma_isolation", "gprec_topology", "deleteability", "overall_dharma")

    def __init__(
        self,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.15,
        cooldown: float = 60.0,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cooldown = cooldown
        self._prev: dict[str, float] = {}
        self._last_alert_time: dict[str, float] = {}
        self._alerts: collections.deque[TopologyAlert] = collections.deque(maxlen=100)

    def check(self, telemetry: DharmaTelemetry) -> list[TopologyAlert]:
        """Check telemetry against previous values and return any new alerts."""
        now = time.time()
        new_alerts: list[TopologyAlert] = []
        for pillar in self.PILLARS:
            val = getattr(telemetry, pillar)
            if pillar not in self._prev:
                self._prev[pillar] = val
                continue
            delta = val - self._prev[pillar]
            abs_delta = abs(delta)
            self._prev[pillar] = val

            if abs_delta < self.warning_threshold:
                continue

            # Check cooldown
            if now - self._last_alert_time.get(pillar, 0) < self.cooldown:
                continue

            direction = "improved" if delta > 0 else "degraded"
            if abs_delta >= self.critical_threshold:
                severity = "critical"
            else:
                severity = "warning"

            label = pillar.replace("_", " ").title()
            alert = TopologyAlert(
                timestamp=now,
                pillar=pillar,
                previous_value=self._prev[pillar] - delta,  # the value before update
                current_value=val,
                delta=delta,
                severity=severity,
                message=f"{label} {direction} by {abs_delta:.3f} ({severity})",
            )
            new_alerts.append(alert)
            self._alerts.append(alert)
            self._last_alert_time[pillar] = now

        return new_alerts

    @property
    def recent_alerts(self) -> list[TopologyAlert]:
        return list(self._alerts)

    def to_json(self, last_n: int = 20) -> list[dict]:
        items = list(self._alerts)[-last_n:]
        return [
            {
                "timestamp": a.timestamp,
                "pillar": a.pillar,
                "previous_value": a.previous_value,
                "current_value": a.current_value,
                "delta": a.delta,
                "severity": a.severity,
                "message": a.message,
            }
            for a in items
        ]


# ===================================================================
# Energy term subclasses (for UniversalDharmaEngine integration)
# ===================================================================


class AniccaTerm(DharmaEnergyTerm):
    """Anicca (Impermanence / Delete-ability) — penalise hard-to-remove nodes.

    Linear energy term: nodes with high removal impact get positive h (penalty).
    """

    def __init__(self, adj: sparse.csr_matrix, weight: float = 1.0):
        super().__init__(weight)
        self.adj = adj

    @property
    def name(self) -> str:
        return "Anicca (Delete-ability)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        _, per_node = compute_deleteability(self.adj)
        # Penalty for low delete-ability
        h = self.weight * (1.0 - per_node)
        if len(h) < n:
            h = np.concatenate([h, np.zeros(n - len(h))])
        elif len(h) > n:
            h = h[:n]
        return h, None


class AnattaTerm(DharmaEnergyTerm):
    """Anatta (Non-self / Karma Isolation) — penalise cross-boundary coupling.

    Submodular: co-selecting nodes with shared side-effect boundaries
    incurs a penalty proportional to their coupling strength.
    """

    def __init__(
        self,
        adj: sparse.csr_matrix,
        boundary_mask: np.ndarray | None = None,
        weight: float = 0.5,
    ):
        super().__init__(weight)
        self.adj = adj
        self.boundary_mask = boundary_mask

    @property
    def name(self) -> str:
        return "Anatta (Karma Isolation)"

    @property
    def math_property(self) -> MathProperty:
        return "submodular"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        adj_n = _resize_sparse(self.adj, n)
        J = _csr_scale_data(adj_n, self.weight)
        return np.zeros(n), J


class PratityaTerm(DharmaEnergyTerm):
    """Pratitya (Dependent Origination / G_prec Topology) — reward modular nodes.

    Linear: nodes in sparse, modular regions of the dependency graph are
    rewarded (negative h = lower energy).
    """

    def __init__(self, adj: sparse.csr_matrix, weight: float = 1.0):
        super().__init__(weight)
        self.adj = adj

    @property
    def name(self) -> str:
        return "Pratitya (Topology Modularity)"

    @property
    def math_property(self) -> MathProperty:
        return "linear"

    def build(self, n: int) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        adj_n = _resize_sparse(self.adj, n)
        out_deg = np.asarray(np.abs(adj_n).sum(axis=1)).flatten()
        in_deg = np.asarray(np.abs(adj_n).sum(axis=0)).flatten()
        degrees = out_deg + in_deg
        max_deg = max(float(degrees.max()), 1e-10)
        # Reward low-degree (modular) nodes
        h = -self.weight * (1.0 - degrees / max_deg)
        return h, None


# ===================================================================
# Engine integration bridge
# ===================================================================


def evaluate_engine_result(
    result,
    evaluator: TopologyEvaluator | None = None,
) -> DharmaTelemetry:
    """Compute topology telemetry from an EngineResult's dependency matrix.

    Parameters
    ----------
    result : EngineResult
        Output from UniversalDharmaEngine.synthesize_and_solve().
    evaluator : TopologyEvaluator | None
        Custom evaluator; uses defaults if None.

    Returns
    -------
    DharmaTelemetry
    """
    if evaluator is None:
        evaluator = TopologyEvaluator()
    if hasattr(result, "J_total") and result.J_total is not None and result.J_total.nnz > 0:
        adj = result.J_total
    else:
        n = len(result.h_total) if hasattr(result, "h_total") else 0
        adj = sparse.csr_matrix(([], ([], [])), shape=(n, n))
    return evaluator.evaluate(adj)
