from __future__ import annotations

"""
Amenity / economic complexity utilities.

This module provides:
  - long POI data -> unit x category counts
  - RCA (Balassa) -> binary specialization matrix M
  - diversity/ubiquity (row/column sums of M)
  - complexity scores via:
      * "juhasz": 2nd mode of raw co-occurrence (SVD on M)
      * "hidalgo": ECI/PCI-style normalized operator (SVD on D^{-1/2} M U^{-1/2})

Design goals:
  - deterministic sign orientation
  - robust SVD behavior for tiny matrices
  - clean, typed public API returning a stable ComplexityProfile object

References (conceptual lineage):
  - Juhász et al. (amenity complexity): RCA->binary M; 2nd eigenvector of similarity
  - Hidalgo & Hausmann (ECI/PCI): method of reflections / normalized operator
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
except Exception as e:  # pragma: no cover
    raise ImportError(
        "core.py requires SciPy (scipy.sparse + scipy.sparse.linalg.svds). "
        "Install with: pip install scipy"
    ) from e


# -----------------------------
# Types
# -----------------------------

Method = Literal["juhasz", "hidalgo"]
Orient = Literal["none", "diversity", "ubiquity"]  # extensible later


@dataclass(frozen=True)
class MatrixBundle:
    """
    A labeled matrix container.

    Use this when you want to carry around a matrix plus its row/col labels
    and a little metadata without losing alignment.
    """
    X: Union[np.ndarray, "csr_matrix"]
    rows: pd.Index
    cols: pd.Index
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ComplexityProfile:
    """
    A compact, stable object holding everything produced by the complexity pipeline.

    Design goal:
      - easy to inspect in notebooks
      - stable fields (so refactors don't break users)
      - supports multiple methods without proliferating return dict keys
    """
    # Core matrices
    counts: Optional[pd.DataFrame] = None        # unit x category counts
    rca: Optional[pd.DataFrame] = None           # continuous RCA
    M: Optional[pd.DataFrame] = None             # binary specialization matrix

    # Metrics tables
    units: Optional[pd.DataFrame] = None         # per-unit metrics (diversity, complexity scores, etc.)
    categories: Optional[pd.DataFrame] = None    # per-category metrics (ubiquity, complexity scores, etc.)

    # Diagnostics / settings
    methods: Tuple[Method, ...] = ("juhasz", "hidalgo")
    params: Dict[str, object] = field(default_factory=dict)
    notes: Dict[str, object] = field(default_factory=dict)

    def copy(self) -> "ComplexityProfile":
        """Convenience deep-ish copy for notebook tinkering."""
        return ComplexityProfile(
            counts=None if self.counts is None else self.counts.copy(),
            rca=None if self.rca is None else self.rca.copy(),
            M=None if self.M is None else self.M.copy(),
            units=None if self.units is None else self.units.copy(),
            categories=None if self.categories is None else self.categories.copy(),
            methods=self.methods,
            params=dict(self.params),
            notes=dict(self.notes),
        )


# -----------------------------
# Public API: matrix building
# -----------------------------

def count_matrix(
    df: pd.DataFrame,
    *,
    unit_col: str,
    category_col: str,
    weight_col: Optional[str] = None,
    fill_value: int = 0,
    min_unit_total: int = 1,
    min_category_total: int = 1,
) -> pd.DataFrame:
    """
    Build a unit x category count (or weighted sum) matrix from long POI data.

    Parameters
    ----------
    df:
        Long dataframe containing at least [unit_col, category_col].
    unit_col, category_col:
        Column names that define the bipartite incidence.
    weight_col:
        If provided, sum weights instead of counting rows.
    min_unit_total, min_category_total:
        Light pruning thresholds applied to the resulting matrix.

    Returns
    -------
    pd.DataFrame
        Wide matrix indexed by unit, columns by category.
    """
    if unit_col not in df.columns or category_col not in df.columns:
        missing = [c for c in (unit_col, category_col) if c not in df.columns]
        raise KeyError(f"Missing required column(s): {missing}")

    if weight_col is None:
        cm = df.groupby([unit_col, category_col]).size().unstack(fill_value=fill_value)
    else:
        if weight_col not in df.columns:
            raise KeyError(f"Missing weight_col='{weight_col}' in df.columns.")
        # Coerce to numeric; non-numeric become NaN then treated as 0 in sum if fill_value is 0.
        w = pd.to_numeric(df[weight_col], errors="coerce")
        tmp = df[[unit_col, category_col]].copy()
        tmp["_w"] = w
        cm = (
            tmp.pivot_table(index=unit_col, columns=category_col, values="_w", aggfunc="sum", fill_value=fill_value)
            .astype(float)
        )

    # Stable ordering helps reproducibility
    cm = cm.sort_index().sort_index(axis=1)

    # Light pruning: drop very small rows/cols; apply until stable (cheap, avoids edge effects)
    cm = _prune_min_totals(cm, min_unit_total=min_unit_total, min_category_total=min_category_total)

    # Ensure integer-like counts stay as ints when possible
    # (weighted matrices will remain float)
    if weight_col is None:
        cm = cm.astype(int)

    return cm


def rca(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revealed comparative advantage (RCA) from a count matrix.

    RCA_{u,c} = (P_{u,c}/P_{u,.}) / (P_{.,c}/P_{..})
              = P_{u,c} * P_{..} / (P_{u,.} * P_{.,c})

    Returns a continuous RCA matrix with the same shape and labels as `counts`.
    """
    if counts.empty:
        return counts.copy()

    X = counts.to_numpy(dtype=float)
    total = X.sum()
    if total <= 0:
        return pd.DataFrame(np.zeros_like(X, dtype=float), index=counts.index, columns=counts.columns)

    r = X.sum(axis=1)  # row sums
    c = X.sum(axis=0)  # col sums
    denom = np.outer(r, c)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(X * total, denom, out=np.zeros_like(X), where=denom != 0)

    out[~np.isfinite(out)] = 0.0
    return pd.DataFrame(out, index=counts.index, columns=counts.columns)


def specialization(
    rca: pd.DataFrame,
    *,
    threshold: float = 1.0,
    dtype: str = "int8",
) -> pd.DataFrame:
    """
    Binarize an RCA matrix to the specialization matrix M (RCA >= threshold).
    """
    if rca.empty:
        return rca.copy()
    return (rca >= threshold).astype(dtype)


def diversity(M: pd.DataFrame) -> pd.Series:
    """
    Row sums of M (units' RCA-diversity).
    """
    return M.sum(axis=1)


def ubiquity(M: pd.DataFrame) -> pd.Series:
    """
    Column sums of M (categories' RCA-ubiquity).
    """
    return M.sum(axis=0)


# -----------------------------
# Public API: complexity methods
# -----------------------------

@overload
def complexity(
    M: pd.DataFrame,
    *,
    method: Method = "juhasz",
    orient: Orient = "diversity",
    zscore: bool = True,
    return_category_scores: Literal[True] = True,
) -> Tuple[pd.Series, pd.Series]: ...


@overload
def complexity(
    M: pd.DataFrame,
    *,
    method: Method = "juhasz",
    orient: Orient = "diversity",
    zscore: bool = True,
    return_category_scores: Literal[False] = False,
) -> pd.Series: ...


def complexity(
    M: pd.DataFrame,
    *,
    method: Method = "juhasz",
    orient: Orient = "diversity",
    zscore: bool = True,
    return_category_scores: bool = True,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Compute complexity scores from a specialization matrix.

    Parameters
    ----------
    M:
        Binary specialization matrix (units x categories).
    method:
        - "juhasz": similarity-based second mode (raw co-occurrence SVD on M)
        - "hidalgo": ECI/PCI-style normalized operator (SVD on D^{-1/2} M U^{-1/2})
    orient:
        Controls deterministic sign choice (to avoid random flips).
          - "diversity": unit scores correlate + with diversity; category scores correlate - with ubiquity
          - "ubiquity": unit scores correlate + with (-avg_ubiquity); category scores correlate - with ubiquity
          - "none": no sign fixing
    zscore:
        If True, standardize scores (mean 0, sd 1) on computed (non-null) entries.
    return_category_scores:
        If True, also return category-side scores.

    Returns
    -------
    pd.Series or (pd.Series, pd.Series)
        Unit complexity, and optionally category complexity.
    """
    method = _normalize_method(method)
    if method == "juhasz":
        unit_s, cat_s = complexity_juhasz(M, orient=orient, zscore=zscore)
    elif method == "hidalgo":
        unit_s, cat_s = complexity_hidalgo(M, orient=orient, zscore=zscore)
    else:  # pragma: no cover
        raise ValueError(f"Unknown method: {method}")

    if return_category_scores:
        return unit_s, cat_s
    return unit_s


def complexity_juhasz(
    M: pd.DataFrame,
    *,
    orient: Orient = "diversity",
    zscore: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Implementation-specific Juhász-style complexity (unit, category).

    Operationally:
      - treat unit similarity ~ M M^T
      - treat category similarity ~ M^T M
      - take the 2nd eigenvector of each (equivalently: 2nd singular vectors of M)

    Notes
    -----
    This is the "raw co-occurrence" version (no degree normalization). If you want
    the classic ECI/PCI normalization, use `complexity_hidalgo`.
    """
    if M.empty:
        return (
            pd.Series(dtype=float, index=M.index, name="complexity_juhasz"),
            pd.Series(dtype=float, index=M.columns, name="complexity_juhasz"),
        )

    # prune all-zero rows/cols for stability and to avoid svd on degenerate structure
    row_mask = (M.sum(axis=1) > 0).to_numpy()
    col_mask = (M.sum(axis=0) > 0).to_numpy()

    M2 = M.loc[M.index[row_mask], M.columns[col_mask]]
    loc_index, cat_index = M2.index, M2.columns

    loc_out = pd.Series(np.nan, index=M.index, name="complexity_juhasz")
    cat_out = pd.Series(np.nan, index=M.columns, name="complexity_juhasz")

    if M2.shape[0] < 2 or M2.shape[1] < 2:
        return loc_out, cat_out

    Ms = csr_matrix(M2.to_numpy(dtype=np.float32))
    svd = _top2_svd(Ms)
    if svd is None:
        return loc_out, cat_out
    U, S, VT = svd

    # second mode: index 1 (index 0 is the dominant trivial mode)
    j = 1 if U.shape[1] > 1 else 0
    unit_scores = U[:, j]
    cat_scores = VT.T[:, j]

    if zscore:
        unit_scores = _zscore(unit_scores)
        cat_scores = _zscore(cat_scores)

    unit_scores, cat_scores = _apply_orientation(
        unit_scores=unit_scores,
        category_scores=cat_scores,
        M=M2,
        orient=orient,
        # On the unit side, "diversity" is the natural reference.
        # On the category side, "ubiquity" is the natural reference.
        # (Complex categories are non-ubiquitous.)
    )

    loc_out.loc[loc_index] = unit_scores
    cat_out.loc[cat_index] = cat_scores
    return loc_out, cat_out


def complexity_hidalgo(
    M: pd.DataFrame,
    *,
    orient: Orient = "diversity",
    zscore: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    ECI/PCI-style complexity (unit, category).

    Compute on normalized matrix:
      B = D^{-1/2} M U^{-1/2}
    then take second singular vectors of B, and map back:
      eci ~ D^{-1/2} u2
      pci ~ U^{-1/2} v2
    """
    if M.empty:
        return (
            pd.Series(dtype=float, index=M.index, name="complexity_hidalgo"),
            pd.Series(dtype=float, index=M.columns, name="complexity_hidalgo"),
        )

    row_mask = (M.sum(axis=1) > 0).to_numpy()
    col_mask = (M.sum(axis=0) > 0).to_numpy()

    M2 = M.loc[M.index[row_mask], M.columns[col_mask]]
    loc_index, cat_index = M2.index, M2.columns

    loc_out = pd.Series(np.nan, index=M.index, name="complexity_hidalgo")
    cat_out = pd.Series(np.nan, index=M.columns, name="complexity_hidalgo")

    if M2.shape[0] < 2 or M2.shape[1] < 2:
        return loc_out, cat_out

    X = M2.to_numpy(dtype=float)
    kc0 = X.sum(axis=1)
    kp0 = X.sum(axis=0)

    inv_sqrt_kc = np.divide(1.0, np.sqrt(kc0), out=np.zeros_like(kc0), where=kc0 > 0)
    inv_sqrt_kp = np.divide(1.0, np.sqrt(kp0), out=np.zeros_like(kp0), where=kp0 > 0)

    B = csr_matrix(X)
    B = B.multiply(inv_sqrt_kc.reshape(-1, 1))
    B = B.multiply(inv_sqrt_kp.reshape(1, -1))

    svd = _top2_svd(B)
    if svd is None:
        return loc_out, cat_out
    U, S, VT = svd

    j = 1 if U.shape[1] > 1 else 0
    eci = inv_sqrt_kc * U[:, j]
    pci = inv_sqrt_kp * VT.T[:, j]

    if zscore:
        eci = _zscore(eci)
        pci = _zscore(pci)

    # Orientation uses the ORIGINAL degree vectors (kc0, kp0) from the pruned matrix.
    if orient != "none":
        if orient == "diversity":
            eci = _orient_sign(eci, kc0, want="positive")
            pci = _orient_sign(pci, kp0, want="negative")
        elif orient == "ubiquity":
            # unit reference: negative average ubiquity of its specialized categories
            # avg_ubiq_u = (M_u* kp0)/kc0
            denom = np.maximum(kc0, 1.0)
            avg_ubiq = (X @ kp0) / denom
            eci = _orient_sign(eci, -avg_ubiq, want="positive")
            pci = _orient_sign(pci, kp0, want="negative")
        else:
            raise ValueError(f"Unknown orient: {orient}")

    loc_out.loc[loc_index] = eci
    cat_out.loc[cat_index] = pci
    return loc_out, cat_out


# -----------------------------
# API
# -----------------------------

def compute_complexity(
    df: pd.DataFrame,
    *,
    unit_col: str = "h3",
    category_col: str = "category",
    weight_col: Optional[str] = None,
    rca_threshold: float = 1.0,
    min_unit_total: int = 1,
    min_category_total: int = 1,
    methods: Sequence[Method] = ("juhasz", "hidalgo"),
    orient: Orient = "diversity",
    zscore: bool = True,
) -> ComplexityProfile:
    """
    End-to-end pipeline:

      df -> counts -> RCA -> M -> diversity/ubiquity -> complexity scores -> ComplexityProfile

    Returns a ComplexityProfile with:
      - counts, rca, M
      - units table: n_amenities, n_categories, diversity_rca, complexity_* columns
      - categories table: ubiquity_rca, complexity_* columns
    """
    methods_t = _normalize_methods(methods)

    # Build matrices
    counts = count_matrix(
        df,
        unit_col=unit_col,
        category_col=category_col,
        weight_col=weight_col,
        min_unit_total=min_unit_total,
        min_category_total=min_category_total,
    )

    rca_mat = rca(counts)
    M = specialization(rca_mat, threshold=rca_threshold)

    # Base metrics
    div = diversity(M).rename("diversity_rca")
    ubi = ubiquity(M).rename("ubiquity_rca")

    units = pd.DataFrame(
        {
            "n_amenities": counts.sum(axis=1),
            "n_categories": (counts > 0).sum(axis=1),
            "diversity_rca": div,
        },
        index=counts.index,
    )

    categories = pd.DataFrame(
        {
            "ubiquity_rca": ubi,
        },
        index=counts.columns,
    )

    # Complexity scores
    for m in methods_t:
        u_s, c_s = complexity(M, method=m, orient=orient, zscore=zscore, return_category_scores=True)
        col = f"complexity_{m}"
        units[col] = u_s
        categories[col] = c_s

    profile = ComplexityProfile(
        counts=counts,
        rca=rca_mat,
        M=M,
        units=units,
        categories=categories,
        methods=methods_t,
        params={
            "unit_col": unit_col,
            "category_col": category_col,
            "weight_col": weight_col,
            "rca_threshold": float(rca_threshold),
            "min_unit_total": int(min_unit_total),
            "min_category_total": int(min_category_total),
            "orient": orient,
            "zscore": bool(zscore),
            "methods": methods_t,
        },
        notes={
            "n_units": int(counts.shape[0]),
            "n_categories": int(counts.shape[1]),
            "n_nonzero_M": int(M.to_numpy(dtype=int).sum()) if not M.empty else 0,
        },
    )
    return profile


# -----------------------------
# Internal utilities
# -----------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize a vector (helper)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - mu) / sd


def _orient_sign(vec: np.ndarray, reference: np.ndarray, want: Literal["positive", "negative"]) -> np.ndarray:
    """
    Flip sign to ensure correlation with reference has desired sign.
    If correlation is nan/undefined, do nothing.
    """
    v = np.asarray(vec, dtype=float)
    r = np.asarray(reference, dtype=float)
    mask = np.isfinite(v) & np.isfinite(r)
    if mask.sum() < 2:
        return v
    v2 = v[mask]
    r2 = r[mask]
    if v2.std() == 0 or r2.std() == 0:
        return v

    corr = np.corrcoef(v2, r2)[0, 1]
    if not np.isfinite(corr):
        return v

    if want == "positive" and corr < 0:
        return -v
    if want == "negative" and corr > 0:
        return -v
    return v


def _prune_zero_rows_cols(M: pd.DataFrame) -> pd.DataFrame:
    """Drop all-zero rows and columns (helper)."""
    if M.empty:
        return M
    rows = M.sum(axis=1) > 0
    cols = M.sum(axis=0) > 0
    return M.loc[rows, cols]


def _top2_svd(A: "csr_matrix") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return the top-2 singular triplets of A in descending singular value order.

    Handles the SciPy constraint k < min(m, n) by falling back to dense SVD when
    min(m, n) <= 2.

    Returns
    -------
    (U, S, VT) or None
      U: (m, k)
      S: (k,)
      VT: (k, n)
    """
    m, n = A.shape
    min_dim = min(m, n)

    if min_dim < 2:
        return None

    if min_dim <= 2:
        U, S, VT = np.linalg.svd(A.toarray(), full_matrices=False)
        # S already descending
        return U, S, VT

    U, S, VT = svds(A, k=2, which="LM")
    order = np.argsort(S)[::-1]
    return U[:, order], S[order], VT[order, :]


def _apply_orientation(
    *,
    unit_scores: np.ndarray,
    category_scores: np.ndarray,
    M: pd.DataFrame,
    orient: Orient,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply deterministic sign orientation to a (unit_scores, category_scores) pair.
    """
    if orient == "none":
        return unit_scores, category_scores

    div = M.sum(axis=1).to_numpy(dtype=float)
    ubi = M.sum(axis=0).to_numpy(dtype=float)
    X = M.to_numpy(dtype=float)

    if orient == "diversity":
        unit_scores = _orient_sign(unit_scores, div, want="positive")
        category_scores = _orient_sign(category_scores, ubi, want="negative")
        return unit_scores, category_scores

    if orient == "ubiquity":
        # unit reference: negative average ubiquity of its specialized categories
        denom = np.maximum(div, 1.0)
        avg_ubiq = (X @ ubi) / denom
        unit_scores = _orient_sign(unit_scores, -avg_ubiq, want="positive")
        category_scores = _orient_sign(category_scores, ubi, want="negative")
        return unit_scores, category_scores

    raise ValueError(f"Unknown orient: {orient}")


def _prune_min_totals(counts: pd.DataFrame, *, min_unit_total: int, min_category_total: int) -> pd.DataFrame:
    """
    Iteratively prune rows/cols below minimum totals until stable.

    This avoids the mild "ping-pong" effect where dropping columns causes some rows
    to fall below the row threshold (and vice versa).
    """
    if counts.empty:
        return counts

    if min_unit_total < 1:
        min_unit_total = 1
    if min_category_total < 1:
        min_category_total = 1

    cm = counts
    while True:
        row_mask = cm.sum(axis=1) >= min_unit_total
        col_mask = cm.sum(axis=0) >= min_category_total
        cm2 = cm.loc[row_mask, col_mask]
        if cm2.shape == cm.shape:
            return cm2
        cm = cm2


def _normalize_method(method: Method) -> Method:
    if method not in ("juhasz", "hidalgo"):
        raise ValueError(f"method must be one of ('juhasz','hidalgo'), got: {method!r}")
    return method


def _normalize_methods(methods: Sequence[Method]) -> Tuple[Method, ...]:
    # preserve order while removing duplicates
    seen = set()
    out: list[Method] = []
    for m in methods:
        m2 = _normalize_method(m)
        if m2 not in seen:
            out.append(m2)
            seen.add(m2)
    if not out:
        out = ["juhasz"]
    return tuple(out)
