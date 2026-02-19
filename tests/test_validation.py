"""Tests for input validation, type safety, cache correctness, and edge cases."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lmm._validation import (
    validate_array_finite,
    validate_k,
    validate_nonneg,
    warn_k_clamped,
)
from lmm.core import LMM
from lmm.qubo import QUBOBuilder
from lmm.solvers import ClassicalQUBOSolver
from lmm.surprise import SurpriseCalculator


# ===================================================================
# _validation module unit tests
# ===================================================================


class TestValidateArrayFinite:
    def test_clean_array_passes(self):
        validate_array_finite(np.array([1.0, 2.0, 3.0]), "test")

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_array_finite(np.array([1.0, float("nan"), 3.0]), "test")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            validate_array_finite(np.array([1.0, float("inf"), 3.0]), "test")

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            validate_array_finite(np.array([1.0, -float("inf"), 3.0]), "test")

    def test_empty_array_passes(self):
        validate_array_finite(np.array([]), "test")

    def test_2d_with_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_array_finite(np.array([[1.0, float("nan")], [3.0, 4.0]]), "test")


class TestValidateK:
    def test_valid_k(self):
        validate_k(1)
        validate_k(100)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_k(0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            validate_k(-5)


class TestValidateNonneg:
    def test_positive_passes(self):
        validate_nonneg(1.0, "alpha")

    def test_zero_passes(self):
        validate_nonneg(0.0, "alpha")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_nonneg(-0.1, "alpha")


class TestWarnKClamped:
    def test_k_within_n(self):
        result = warn_k_clamped(5, 10)
        assert result == 5

    def test_k_equals_n(self):
        result = warn_k_clamped(10, 10)
        assert result == 10

    def test_k_exceeds_n_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = warn_k_clamped(20, 10)
            assert result == 10
            assert len(w) == 1
            assert "clamping" in str(w[0].message).lower()


# ===================================================================
# LMM constructor validation
# ===================================================================


class TestLMMParameterValidation:
    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            LMM(k=0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LMM(k=5, alpha=-1.0)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LMM(k=5, gamma=-10.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LMM(k=5, beta=-0.5)

    def test_zero_alpha_allowed(self):
        model = LMM(k=5, alpha=0.0)
        assert model.alpha == 0.0

    def test_zero_gamma_allowed(self):
        model = LMM(k=5, gamma=0.0)
        assert model.gamma == 0.0


# ===================================================================
# NaN/Inf input tests
# ===================================================================


class TestNaNInfInputs:
    def test_lmm_select_nan_candidates(self):
        model = LMM(k=3)
        model.fit(np.random.RandomState(42).randn(50))
        with pytest.raises(ValueError, match="NaN"):
            model.select(np.array([1.0, float("nan"), 3.0]))

    def test_lmm_select_inf_candidates(self):
        model = LMM(k=3)
        model.fit(np.random.RandomState(42).randn(50))
        with pytest.raises(ValueError, match="Inf"):
            model.select(np.array([1.0, float("inf"), 3.0]))

    def test_lmm_select_from_surprises_nan(self):
        model = LMM(k=3)
        with pytest.raises(ValueError, match="NaN"):
            model.select_from_surprises(np.array([1.0, float("nan"), 3.0]))

    def test_surprise_fit_nan_raises(self):
        calc = SurpriseCalculator(method="kl")
        with pytest.raises(ValueError, match="NaN"):
            calc.fit(np.array([1.0, float("nan"), 3.0]))

    def test_surprise_compute_nan_raises(self):
        calc = SurpriseCalculator(method="kl")
        calc.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="NaN"):
            calc.compute(np.array([float("nan")]))

    def test_similarity_matrix_nan_raises(self):
        model = LMM(k=2, beta=1.0)
        model.fit(np.random.RandomState(42).randn(10))
        sim = np.eye(5)
        sim[0, 1] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            model.select(
                np.random.RandomState(42).randn(5),
                similarity_matrix=sim,
            )


# ===================================================================
# k > n warning tests
# ===================================================================


class TestKClampWarning:
    def test_lmm_select_k_exceeds_n(self):
        model = LMM(k=20)
        model.fit(np.random.RandomState(42).randn(50))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.select(np.array([1.0, 2.0, 3.0]))
            assert len(result.selected_indices) <= 3
            assert any("clamping" in str(x.message).lower() for x in w)

    def test_lmm_select_from_surprises_k_exceeds_n(self):
        model = LMM(k=20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.select_from_surprises(np.array([1.0, 2.0, 3.0]))
            assert len(result.selected_indices) <= 3
            assert any("clamping" in str(x.message).lower() for x in w)


# ===================================================================
# QUBO shape validation tests
# ===================================================================


class TestQUBOValidation:
    def test_surprise_length_mismatch_raises(self):
        builder = QUBOBuilder(n_variables=5)
        with pytest.raises(ValueError, match="length"):
            builder.add_surprise_objective(np.array([1.0, 2.0, 3.0]))

    def test_similarity_shape_mismatch_raises(self):
        builder = QUBOBuilder(n_variables=5)
        with pytest.raises(ValueError, match="shape"):
            builder.add_diversity_penalty(np.eye(3))

    def test_cardinality_k_out_of_range_raises(self):
        builder = QUBOBuilder(n_variables=5)
        with pytest.raises(ValueError, match="out of range"):
            builder.add_cardinality_constraint(k=10, gamma=10.0)

    def test_cardinality_negative_k_raises(self):
        builder = QUBOBuilder(n_variables=5)
        with pytest.raises(ValueError, match="out of range"):
            builder.add_cardinality_constraint(k=-1, gamma=10.0)


# ===================================================================
# Cache invalidation correctness tests
# ===================================================================


class TestQUBOCacheInvalidation:
    def test_cache_after_surprise_then_diversity(self):
        """Ensure cache is correct when adding surprise, then diversity."""
        builder = QUBOBuilder(n_variables=5)
        surprises = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        builder.add_surprise_objective(surprises)
        builder.add_cardinality_constraint(k=2, gamma=10.0)

        # Force cache build
        e1 = builder.evaluate(np.array([1, 0, 0, 1, 0], dtype=float))

        # Now add diversity penalty (modifies off-diagonal)
        sim = np.ones((5, 5)) * 0.1
        np.fill_diagonal(sim, 0)
        builder.add_diversity_penalty(sim, beta=1.0)

        # Energy should change
        e2 = builder.evaluate(np.array([1, 0, 0, 1, 0], dtype=float))
        assert e1 != e2

    def test_cache_consistent_with_dense(self):
        """evaluate() via sparse cache must match dense x^T Q x."""
        builder = QUBOBuilder(n_variables=6)
        rng = np.random.RandomState(42)
        builder.add_surprise_objective(rng.rand(6))
        builder.add_cardinality_constraint(k=2, gamma=10.0)
        sim = rng.rand(6, 6) * 0.2
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0)
        builder.add_diversity_penalty(sim, beta=0.5)

        x = np.array([1, 0, 1, 0, 0, 0], dtype=float)
        sparse_energy = builder.evaluate(x)
        dense_energy = float(x @ builder.Q @ x)
        assert abs(sparse_energy - dense_energy) < 1e-10

    def test_repeated_evaluate_stable(self):
        """Multiple evaluate() calls return the same result."""
        builder = QUBOBuilder(n_variables=4)
        builder.add_surprise_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        builder.add_cardinality_constraint(k=2, gamma=5.0)
        x = np.array([0, 1, 1, 0], dtype=float)
        e1 = builder.evaluate(x)
        e2 = builder.evaluate(x)
        e3 = builder.evaluate(x)
        assert e1 == e2 == e3


# ===================================================================
# Method string normalization tests
# ===================================================================


class TestMethodNormalization:
    def test_solver_method_case_insensitive(self):
        """Uppercase method like 'SA' should work with a warning."""
        builder = QUBOBuilder(n_variables=10)
        builder.add_surprise_objective(np.random.RandomState(42).rand(10))
        builder.add_cardinality_constraint(k=3, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = solver.solve(method="SA", k=3)
            assert isinstance(x, np.ndarray)
            assert any("normalized" in str(m.message).lower() for m in w)

    def test_solver_method_whitespace(self):
        """Method with trailing space should work with a warning."""
        builder = QUBOBuilder(n_variables=10)
        builder.add_surprise_objective(np.random.RandomState(42).rand(10))
        builder.add_cardinality_constraint(k=3, gamma=10.0)
        solver = ClassicalQUBOSolver(builder)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = solver.solve(method="sa ", k=3)
            assert isinstance(x, np.ndarray)
            assert any("normalized" in str(m.message).lower() for m in w)

    def test_surprise_method_normalization(self):
        """SurpriseCalculator should normalize method string."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc = SurpriseCalculator(method="KL")
            assert calc.method == "kl"
            assert len(w) == 1


# ===================================================================
# dtype acceptance tests
# ===================================================================


class TestDtypeAcceptance:
    def test_int_array_accepted(self):
        """Integer arrays should be accepted (converted internally)."""
        model = LMM(k=2)
        data = np.array([1, 2, 3, 4, 5])
        model.fit(data)
        result = model.select(data)
        assert len(result.selected_indices) == 2

    def test_float32_accepted(self):
        """float32 arrays should work without error."""
        model = LMM(k=2)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        model.fit(data)
        result = model.select(data)
        assert len(result.selected_indices) == 2
