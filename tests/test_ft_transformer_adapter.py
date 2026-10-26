from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.ft_transformer import FTTransformerAdapter
from src.paths import PROJECT_ROOT


def small_params(**updates):
    return {
        "device": "cpu",
        "n_blocks": 1,
        "d_block": 16,
        "attention_n_heads": 4,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "residual_dropout": 0.0,
        **updates,
    }


def adapter_for(
    features: tuple[str, ...] = ("numeric", "category"),
    categorical_features: tuple[str, ...] = ("category",),
    **params,
) -> FTTransformerAdapter:
    return FTTransformerAdapter(
        features,
        categorical_features,
        "diagnosed_diabetes",
        seed=42,
        model_params=small_params(**params),
    )


def training_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "numeric": np.linspace(-2.0, 2.0, rows, dtype=np.float64),
        "category": ["a", "b", "a", "c", "b", "c", "a", "b"][:rows],
    })


def controlled_untrained_adapter() -> tuple[FTTransformerAdapter, pd.DataFrame]:
    """Create a test-only prediction state with random, untrained weights."""
    train = training_frame()
    adapter = adapter_for()
    adapter._fit_preprocessing(train)
    with adapter._torch.random.fork_rng(devices=[]):
        adapter._torch.random.default_generator.manual_seed(adapter.seed)
        adapter._model = adapter._construct_model()
    adapter._ready = True
    return adapter, train


class FTTransformerPreprocessingTests(unittest.TestCase):
    def test_numeric_statistics_are_training_only_and_transform_is_linear(self):
        train = pd.DataFrame({
            "numeric": [-2.0, 0.0, 2.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "binary": [0.0, 1.0, 0.0, 1.0],
        })
        adapter = adapter_for(
            ("numeric", "constant", "binary"), (), device="cpu",
        )
        adapter._fit_preprocessing(train)
        before = adapter._preprocessing_metadata()

        held_out = pd.DataFrame({
            "numeric": [1_000_000.0],
            "constant": [10.0],
            "binary": [1.0],
        })
        transformed, categories = adapter._transform_inputs(held_out)
        after = adapter._preprocessing_metadata()

        self.assertIsNone(categories)
        np.testing.assert_array_equal(before["numeric_means"], [1.0, 7.0, 0.5])
        np.testing.assert_array_equal(before["numeric_scales"], [np.sqrt(5.0), 1.0, 0.5])
        self.assertEqual(before, after)
        self.assertGreater(transformed[0, 0], 100_000.0)
        self.assertEqual(transformed[0, 1], 3.0)
        self.assertEqual(transformed[0, 2], 1.0)
        train_transformed, _ = adapter._transform_inputs(train)
        np.testing.assert_array_equal(train_transformed[:, 1], np.zeros(len(train)))

    def test_repeated_and_binary_values_are_deterministic(self):
        train = pd.DataFrame({
            "repeated": [2.0] * 4 + [5.0] * 4,
            "indicator": [0.0, 1.0] * 4,
        })
        adapter = adapter_for(("repeated", "indicator"), ())
        adapter._fit_preprocessing(train)
        first, _ = adapter._transform_inputs(train)
        second, _ = adapter._transform_inputs(train.copy(deep=True))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_nonfinite_inputs_and_float32_overflow_are_rejected(self):
        train = pd.DataFrame({"numeric": [0.0, 1.0, 2.0]})
        adapter = adapter_for(("numeric",), ())
        adapter._fit_preprocessing(train)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter._transform_inputs(pd.DataFrame({"numeric": [value]}))
        with self.assertRaisesRegex(ValueError, "non-finite float32"):
            adapter._transform_inputs(pd.DataFrame({"numeric": [1e300]}))

    def test_categorical_vocabulary_mode_fallback_and_diagnostics(self):
        train = pd.DataFrame({
            "numeric": np.arange(5, dtype=np.float64),
            "category": ["z", "a", "z", "a", "m"],
        })
        adapter = adapter_for()
        original = train.copy(deep=True)
        adapter._fit_preprocessing(train)

        self.assertEqual(adapter._category_vocabularies["category"], ("a", "m", "z"))
        self.assertEqual(
            adapter._category_training_counts["category"],
            {"a": 2, "m": 1, "z": 2},
        )
        self.assertEqual(adapter._category_fallback_values["category"], "a")
        self.assertEqual(adapter._category_fallback_indices["category"], 0)

        query = pd.DataFrame({"numeric": [9.0, 10.0], "category": ["new", "z"]})
        query_copy = query.copy(deep=True)
        before = adapter._preprocessing_metadata()
        counts_first = adapter.unseen_category_counts(query)
        counts_second = adapter.unseen_category_counts(query)
        _, encoded = adapter._transform_inputs(query)

        self.assertEqual(counts_first, {"category": 1})
        self.assertEqual(counts_second, counts_first)
        self.assertEqual(encoded[:, 0].tolist(), [0, 2])
        self.assertEqual(adapter._preprocessing_metadata(), before)
        pd.testing.assert_frame_equal(train, original)
        pd.testing.assert_frame_equal(query, query_copy)

    def test_numeric_categories_use_base_string_normalization(self):
        train = pd.DataFrame({"category": [2, 10, 2, 10]})
        adapter = adapter_for(("category",), ("category",))
        adapter._fit_preprocessing(train)
        self.assertEqual(adapter._category_vocabularies["category"], ("10", "2"))
        query = pd.DataFrame({"category": [2, 99]})
        self.assertEqual(adapter.unseen_category_counts(query), {"category": 1})
        _, encoded = adapter._transform_inputs(query)
        self.assertEqual(encoded[:, 0].tolist(), [1, 0])

    def test_unseen_diagnostics_require_fitted_preprocessing(self):
        adapter = adapter_for()
        with self.assertRaisesRegex(RuntimeError, "preprocessing must be fitted"):
            adapter.unseen_category_counts(training_frame())

    def test_unavailable_cuda_fails_without_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                adapter_for(device="cuda:0")

    def test_schema_contract_rejects_missing_extra_duplicate_and_reordered_columns(self):
        adapter = adapter_for()
        adapter._fit_preprocessing(training_frame())
        valid = training_frame()
        bad_frames = (
            valid.drop(columns=["numeric"]),
            valid.assign(extra=1),
            valid.loc[:, ["category", "numeric"]],
            pd.concat([valid, valid[["numeric"]]], axis=1),
        )
        for bad in bad_frames:
            with self.subTest(columns=tuple(bad.columns)), self.assertRaises(ValueError):
                adapter._transform_inputs(bad)


class FTTransformerPredictionTests(unittest.TestCase):
    def test_model_construction_supports_numeric_only_and_categorical_only(self):
        numeric = adapter_for(("numeric",), ())
        numeric_train = pd.DataFrame({"numeric": np.arange(4, dtype=np.float64)})
        numeric._fit_preprocessing(numeric_train)
        numeric_model = numeric._construct_model()
        numerical, categorical = numeric._transform_inputs(numeric_train)
        logits = numeric_model(*numeric._to_tensors(numerical, categorical))
        self.assertEqual(tuple(logits.shape), (4, 1))
        self.assertIsNone(categorical)

        categorical_only = adapter_for(("category",), ("category",))
        categorical_train = pd.DataFrame({"category": ["a", "b", "a", "c"]})
        categorical_only._fit_preprocessing(categorical_train)
        categorical_model = categorical_only._construct_model()
        numerical, categorical = categorical_only._transform_inputs(categorical_train)
        logits = categorical_model(*categorical_only._to_tensors(numerical, categorical))
        self.assertEqual(tuple(logits.shape), (4, 1))
        self.assertIsNone(numerical)
        self.assertEqual(categorical.dtype, np.int64)

    def test_probabilities_order_repetition_batches_and_model_immutability(self):
        adapter, _train = controlled_untrained_adapter()
        query = pd.DataFrame({
            "numeric": np.linspace(-3.0, 3.0, 11),
            "category": ["a", "b", "missing", "c", "a", "b", "c", "a", "missing", "b", "c"],
        })
        query_copy = query.copy(deep=True)
        model = adapter._model
        list(model.modules())[1].eval()
        original_mode = model.training
        module_modes = tuple(module.training for module in model.modules())
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
        preprocessing_before = adapter._preprocessing_metadata()

        predictions = {}
        for batch_size in (1, 3, 8):
            logits = adapter._predict_logits(query, batch_size=batch_size, model=model)
            predictions[batch_size] = adapter._sigmoid_probabilities(logits, len(query))
        public = adapter.predict_proba(query)
        repeated = adapter.predict_proba(query)

        self.assertEqual(public.shape, (11,))
        self.assertEqual(public.dtype, np.float64)
        self.assertTrue(np.isfinite(public).all())
        self.assertTrue(((public >= 0) & (public <= 1)).all())
        np.testing.assert_allclose(public, repeated, rtol=1e-5, atol=1e-6)
        for batch_size in (1, 8):
            np.testing.assert_allclose(predictions[batch_size], predictions[3], rtol=1e-5, atol=1e-6)
        permutation = np.array([10, 4, 0, 9, 3, 1, 8, 7, 5, 2, 6])
        inverse = np.argsort(permutation)
        permuted = adapter.predict_proba(query.iloc[permutation])
        np.testing.assert_allclose(permuted[inverse], public, rtol=1e-5, atol=1e-6)
        pd.testing.assert_frame_equal(query, query_copy)
        self.assertEqual(adapter._preprocessing_metadata(), preprocessing_before)
        self.assertEqual(model.training, original_mode)
        self.assertEqual(tuple(module.training for module in model.modules()), module_modes)
        for parameter, before in zip(model.parameters(), parameters_before):
            self.assertTrue(self._torch_equal(parameter, before))

    @staticmethod
    def _torch_equal(left, right):
        import torch

        return torch.equal(left.detach(), right)

    def test_nonfinite_logits_fail_and_restore_model_mode(self):
        adapter, _ = controlled_untrained_adapter()
        model = adapter._model
        model.train()
        list(model.modules())[1].eval()
        module_modes = tuple(module.training for module in model.modules())
        query = training_frame()
        torch = adapter._torch
        with patch.object(model, "forward", return_value=torch.full((len(query), 1), float("inf"))):
            with self.assertRaisesRegex(ValueError, "non-finite logits"):
                adapter.predict_proba(query)
        self.assertTrue(model.training)
        self.assertEqual(tuple(module.training for module in model.modules()), module_modes)

    def test_prediction_scopes_backend_settings_for_trained_and_loaded_adapters(self):
        train = training_frame()
        fitted = adapter_for(max_epochs=1, patience=1)
        with redirect_stdout(io.StringIO()):
            fitted.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / fitted.artifact_name
            fitted.save(path)
            adapters = (fitted, FTTransformerAdapter.load(path, device="cpu"))

        torch = fitted._torch
        backend = torch.backends

        def backend_state():
            return (
                backend.cuda.matmul.allow_tf32,
                backend.cudnn.allow_tf32,
                backend.cudnn.benchmark,
                backend.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )

        original = backend_state()
        original_rng = torch.random.get_rng_state().clone()
        try:
            backend.cuda.matmul.allow_tf32 = True
            backend.cudnn.allow_tf32 = True
            backend.cudnn.benchmark = True
            backend.cudnn.deterministic = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            caller_settings = backend_state()
            expected_profile = (False, False, False, True, False)

            for adapter in adapters:
                model = adapter._model
                observed_settings = []

                def forward(_x_cont, _x_cat):
                    observed_settings.append(backend_state()[:5])
                    return torch.zeros((len(train), 1), dtype=torch.float32)

                with patch.object(model, "forward", side_effect=forward):
                    probabilities = adapter.predict_proba(train)
                self.assertEqual(probabilities.shape, (len(train),))
                self.assertEqual(observed_settings, [expected_profile])
                self.assertEqual(backend_state(), caller_settings)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))

                with patch.object(
                    model, "forward", side_effect=RuntimeError("inference failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "inference failure"):
                        adapter.predict_proba(train)
                self.assertEqual(backend_state(), caller_settings)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
        finally:
            backend.cuda.matmul.allow_tf32 = original[0]
            backend.cudnn.allow_tf32 = original[1]
            backend.cudnn.benchmark = original[2]
            backend.cudnn.deterministic = original[3]
            torch.use_deterministic_algorithms(original[4], warn_only=original[5])

    def test_public_lifecycle_is_not_completed_by_preprocessing_or_construction(self):
        adapter = adapter_for()
        train = training_frame()
        adapter._fit_preprocessing(train)
        adapter._model = adapter._construct_model()
        self.assertFalse(adapter.metadata()["ready"])
        with self.assertRaisesRegex(RuntimeError, "complete training"):
            adapter.predict_proba(train)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "complete training"):
                adapter.save(Path(temporary) / "model.ft_transformer.pt")
            self.assertEqual(list(Path(temporary).iterdir()), [])
        with self.assertRaisesRegex(RuntimeError, "refitting is unsupported"):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))

    def test_importing_adapter_does_not_import_optional_torch_or_model_package(self):
        code = """
import sys
import src.models.ft_transformer
assert "torch" not in sys.modules
assert "rtdl_revisiting_models" not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class FTTransformerTrainingTests(unittest.TestCase):
    def test_validation_frequency_must_match_frozen_every_epoch_policy(self):
        with self.assertRaisesRegex(ValueError, "validation_frequency must equal 1"):
            adapter_for(validation_frequency=2)

    def test_label_shape_length_values_and_series_alignment_are_validated(self):
        train = training_frame()
        adapter = adapter_for()
        mismatched_index = pd.Series([0, 1] * 4, index=np.arange(1, 9))
        with self.assertRaisesRegex(ValueError, "index must exactly match"):
            adapter.fit(train, mismatched_index, train, pd.Series([0, 1] * 4))
        self.assertFalse(adapter.metadata()["ready"])

        cases = (
            (np.array([[0, 1] * 4]), "one-dimensional"),
            (np.array([0, 1] * 3), "lengths must match"),
            (np.array([0, 1, 2, 0, 1, 0, 1, 0]), "only 0 and 1"),
            (np.array([0, 1, 0, 1, 0, np.nan, 0, 1]), "finite real numbers"),
            (np.zeros(8), "both 0 and 1"),
        )
        for labels, message in cases:
            with self.subTest(message=message):
                adapter = adapter_for()
                with self.assertRaisesRegex(ValueError, message):
                    adapter.fit(train, labels, train, np.array([0, 1] * 4))
                self.assertFalse(adapter.metadata()["ready"])
                with self.assertRaises(RuntimeError):
                    adapter.predict_proba(train)

    def test_failed_fit_cannot_predict_save_or_refit(self):
        train = training_frame()
        adapter = adapter_for(max_epochs=1)
        with patch.object(adapter, "_train_epoch", side_effect=RuntimeError("synthetic training failure")):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                RuntimeError, "synthetic training failure",
            ):
                adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        self.assertFalse(adapter.metadata()["ready"])
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(train)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                adapter.save(Path(temporary) / adapter.artifact_name)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        with self.assertRaisesRegex(RuntimeError, "refitting is unsupported"):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))

    def test_one_class_validation_is_rejected_before_training(self):
        train = training_frame()
        adapter = adapter_for()
        with self.assertRaisesRegex(ValueError, "validation: labels must contain both 0 and 1"):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0] * 8))
        self.assertIsNone(adapter._model)
        self.assertFalse(adapter.metadata()["ready"])

    def test_exact_fixed_budget_tiny_overfit_diagnostic(self):
        values = np.array([-2.0, -1.0, -2.0, -1.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        train = pd.DataFrame({"numeric": values})
        adapter = adapter_for(
            ("numeric",),
            (),
            n_blocks=1,
            d_block=16,
            attention_n_heads=4,
            attention_dropout=0.0,
            ffn_d_hidden=32,
            ffn_d_hidden_multiplier=None,
            ffn_activation="ReGLU",
            ffn_dropout=0.0,
            residual_dropout=0.0,
            learning_rate=0.01,
            weight_decay=0.0,
            train_batch_size=8,
        )
        adapter._fit_preprocessing(train)
        train_numeric, train_categorical = adapter._transform_inputs(train)
        generator = adapter._torch.Generator(device="cpu")
        generator.manual_seed(43)
        start = time.perf_counter()
        with adapter._torch.random.fork_rng(devices=[]):
            adapter._torch.random.default_generator.manual_seed(42)
            model = adapter._construct_model()
            optimizer = adapter._torch.optim.AdamW(
                model.make_parameter_groups(),
                lr=0.01,
                weight_decay=0.0,
            )
            before = [parameter.detach().clone() for parameter in model.parameters()]
            for _ in range(500):
                adapter._train_epoch(
                    model, optimizer, train_numeric, train_categorical, labels, generator,
                )
            logits = adapter._predict_logits(train, batch_size=8, model=model)
        elapsed = time.perf_counter() - start
        tensor_logits = adapter._torch.as_tensor(logits, dtype=adapter._torch.float32)
        loss = float(adapter._torch.nn.functional.binary_cross_entropy_with_logits(
            tensor_logits, adapter._torch.as_tensor(labels, dtype=adapter._torch.float32),
        ).item())
        accuracy = float(np.mean((logits >= 0) == labels))
        self.assertTrue(any(
            not adapter._torch.equal(previous, current.detach())
            for previous, current in zip(before, model.parameters())
        ))
        self.assertLess(loss, 0.05)
        self.assertEqual(accuracy, 1.0)
        print(
            f"Tiny FT-Transformer overfit diagnostic: epochs=500, "
            f"final_train_bce={loss:.8f}, accuracy={accuracy:.3f}, "
            f"runtime_seconds={elapsed:.3f}",
        )

    def test_checkpoint_ties_small_improvements_patience_and_snapshots(self):
        adapter = adapter_for(patience=2, min_delta=0.01)
        model = adapter._torch.nn.Linear(1, 1, bias=False)
        state = {
            "best_auc": None, "best_epoch": None, "best_state": None,
            "patience_reference": None, "patience_counter": 0,
        }

        with adapter._torch.no_grad():
            model.weight.fill_(1.0)
        self.assertFalse(adapter._record_selection(state, epoch=1, auc=0.6, model=model))
        first_snapshot = state["best_state"]["weight"].clone()
        with adapter._torch.no_grad():
            model.weight.fill_(2.0)
        self.assertFalse(adapter._record_selection(state, epoch=2, auc=0.6, model=model))
        self.assertEqual(state["best_epoch"], 1)
        self.assertTrue(adapter._torch.equal(state["best_state"]["weight"], first_snapshot))
        with adapter._torch.no_grad():
            model.weight.fill_(3.0)
        self.assertTrue(adapter._record_selection(state, epoch=3, auc=0.6005, model=model))
        self.assertEqual(state["best_epoch"], 3)
        self.assertEqual(state["patience_reference"], 0.6)
        self.assertEqual(state["patience_counter"], 2)
        with self.assertRaisesRegex(RuntimeError, "finite validation ROC-AUC"):
            adapter._record_selection(state, epoch=4, auc=float("nan"), model=model)

    def test_patience_reference_uses_cumulative_qualifying_improvement(self):
        adapter = adapter_for(patience=2, min_delta=0.01)
        model = adapter._torch.nn.Linear(1, 1, bias=False)
        state = {
            "best_auc": None, "best_epoch": None, "best_state": None,
            "patience_reference": None, "patience_counter": 0,
        }
        self.assertFalse(adapter._record_selection(state, epoch=1, auc=0.6, model=model))
        self.assertFalse(adapter._record_selection(state, epoch=2, auc=0.605, model=model))
        self.assertEqual(state["patience_counter"], 1)
        self.assertFalse(adapter._record_selection(state, epoch=3, auc=0.6101, model=model))
        self.assertEqual(state["patience_reference"], 0.6101)
        self.assertEqual(state["patience_counter"], 0)

    def test_fit_restores_highest_auc_weights_and_epoch_cap_reason(self):
        train = training_frame()
        adapter = adapter_for(max_epochs=2, patience=5)
        scores = iter((0.8, 0.7))
        updates = iter((1.0, 2.0))

        def fake_train(model, *_args):
            with adapter._torch.no_grad():
                next(model.parameters()).fill_(next(updates))
            return 0.5

        with patch.object(adapter, "_train_epoch", side_effect=fake_train), patch.object(
            adapter, "_evaluate_validation",
            side_effect=lambda *_args: (0.4, next(scores)),
        ), redirect_stdout(io.StringIO()):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))

        metadata = adapter.metadata()["training"]
        self.assertEqual(metadata["best_epoch"], 1)
        self.assertEqual(metadata["best_validation_roc_auc"], 0.8)
        self.assertEqual(metadata["epochs_executed"], 2)
        self.assertEqual(metadata["stopping_reason"], "max_epochs")
        self.assertTrue(adapter._torch.equal(next(adapter._model.parameters()).flatten()[0], adapter._torch.tensor(1.0)))

    def test_patience_stop_takes_precedence_at_epoch_cap(self):
        train = training_frame()
        adapter = adapter_for(max_epochs=2, patience=1, min_delta=0.001)
        scores = iter((0.8, 0.8))
        with patch.object(adapter, "_train_epoch", return_value=0.5), patch.object(
            adapter, "_evaluate_validation",
            side_effect=lambda *_args: (0.4, next(scores)),
        ), redirect_stdout(io.StringIO()):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        self.assertEqual(adapter.metadata()["training"]["stopping_reason"], "patience")

    def test_backend_settings_restore_on_fit_success_and_failure(self):
        train = training_frame()
        torch = adapter_for()._torch
        backends = torch.backends
        original = (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        original_rng = torch.random.get_rng_state().clone()
        try:
            backends.cuda.matmul.allow_tf32 = True
            backends.cudnn.allow_tf32 = True
            backends.cudnn.benchmark = True
            backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            expected = (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            adapter = adapter_for(max_epochs=1)
            with patch.object(adapter, "_train_epoch", return_value=0.5), patch.object(
                adapter, "_evaluate_validation", return_value=(0.4, 0.75),
            ), redirect_stdout(io.StringIO()):
                adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
            actual = (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            self.assertEqual(actual, expected)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))

            failed = adapter_for(max_epochs=1)
            with patch.object(failed, "_train_epoch", side_effect=RuntimeError("failure")), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    failed.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
            restored_after_failure = (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            self.assertEqual(restored_after_failure, expected)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
        finally:
            backends.cuda.matmul.allow_tf32 = original[0]
            backends.cudnn.allow_tf32 = original[1]
            backends.cudnn.benchmark = original[2]
            backends.cudnn.deterministic = original[3]
            torch.use_deterministic_algorithms(original[4], warn_only=original[5])

    def test_cpu_resource_hooks(self):
        adapter = adapter_for()
        adapter.synchronize()
        adapter.reset_peak_memory()
        self.assertEqual(adapter.resource_metadata(), {"peak_gpu_memory": None})

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA is unavailable")
    def test_bounded_cuda_fit_and_resource_hooks(self):
        train = training_frame()
        adapter = adapter_for(device="cuda:0", max_epochs=1, patience=1)
        with redirect_stdout(io.StringIO()):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        adapter.synchronize()
        probabilities = adapter.predict_proba(train.iloc[:3])
        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(np.isfinite(probabilities).all())
        memory = adapter.resource_metadata()["peak_gpu_memory"]
        self.assertEqual(memory["device"], "cuda:0")
        self.assertGreater(memory["allocated_bytes"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            loaded = FTTransformerAdapter.load(path, device="cuda:0")
            np.testing.assert_allclose(
                loaded.predict_proba(train.iloc[:3]), probabilities,
                rtol=1e-5, atol=1e-6,
            )


class FTTransformerPersistenceTests(unittest.TestCase):
    def fitted_adapter(self) -> tuple[FTTransformerAdapter, pd.DataFrame]:
        train = training_frame()
        adapter = adapter_for(max_epochs=2, patience=5)
        with redirect_stdout(io.StringIO()):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        return adapter, train

    def test_fit_save_load_predict_round_trip_without_refitting_preprocessing(self):
        adapter, train = self.fitted_adapter()
        query = pd.DataFrame({
            "numeric": [-3.0, 0.0, 3.0],
            "category": ["unknown", "a", "c"],
        })
        expected = adapter.predict_proba(query)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            with patch(
                "src.models.ft_transformer.StandardScaler.fit",
                side_effect=AssertionError("load must not refit preprocessing"),
            ):
                restored = FTTransformerAdapter.load(path, device="cpu")
            actual = restored.predict_proba(query)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(restored.metadata()["preprocessing"], adapter.metadata()["preprocessing"])
            self.assertEqual(restored.metadata()["training"], adapter.metadata()["training"])
            self.assertEqual(restored.unseen_category_counts(query), {"category": 1})
            self.assertEqual(restored.metadata()["training"]["training_population_size"], len(train))
            self.assertEqual(restored.metadata()["artifact_path"], str(path))

    def test_categorical_features_are_canonicalized_and_round_trip(self):
        features = ("numeric", "second_category", "first_category")
        categorical_features = ("first_category", "second_category")
        train = pd.DataFrame({
            "numeric": np.linspace(-1.0, 1.0, 8),
            "second_category": ["x", "y"] * 4,
            "first_category": ["b", "a", "a", "b"] * 2,
        })
        adapter = adapter_for(
            features,
            categorical_features,
            max_epochs=1,
            patience=1,
        )
        self.assertEqual(
            adapter.categorical_features,
            ("second_category", "first_category"),
        )
        with redirect_stdout(io.StringIO()):
            adapter.fit(train, pd.Series([0, 1] * 4), train, pd.Series([0, 1] * 4))
        expected = adapter.predict_proba(train)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            restored = FTTransformerAdapter.load(path, device="cpu")
        self.assertEqual(
            restored.categorical_features,
            ("second_category", "first_category"),
        )
        np.testing.assert_allclose(
            restored.predict_proba(train), expected, rtol=1e-5, atol=1e-6,
        )

    def test_atomic_save_failure_cleans_temporary_file(self):
        adapter, _ = self.fitted_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            with patch("src.models.ft_transformer.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    adapter.save(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_missing_corrupt_unsupported_and_device_incompatible_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.ft_transformer.pt"
            with self.assertRaises(FileNotFoundError):
                FTTransformerAdapter.load(path, device="cpu")
            path.write_bytes(b"not a torch artifact")
            with self.assertRaisesRegex(ValueError, "Unable to read"):
                FTTransformerAdapter.load(path, device="cpu")

            adapter, _ = self.fitted_adapter()
            adapter.save(path)
            torch = adapter._torch
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["format_version"] = 99
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "format version"):
                FTTransformerAdapter.load(path, device="cpu")

            payload["format_version"] = 1
            with self.assertRaisesRegex(RuntimeError, "Cross-device reload is unsupported"):
                payload["device"] = "cuda:0"
                torch.save(payload, path)
                FTTransformerAdapter.load(path, device="cpu")

    def test_load_rejects_version_preprocessing_and_state_inconsistency(self):
        adapter, _ = self.fitted_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            torch = adapter._torch
            payload = torch.load(path, map_location="cpu", weights_only=True)

            payload["device"] = "cpu"
            payload["versions"]["rtdl-revisiting-models"] = "9.9.9"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "rtdl-revisiting-models version"):
                FTTransformerAdapter.load(path, device="cpu")

            payload["versions"]["rtdl-revisiting-models"] = adapter._versions()["rtdl-revisiting-models"]
            payload["versions"]["torch"] = "0.0.0"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "torch version"):
                FTTransformerAdapter.load(path, device="cpu")

            payload["versions"]["torch"] = adapter._versions()["torch"]
            payload["preprocessing"]["numeric_scales"][0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "numerical means/scales"):
                FTTransformerAdapter.load(path, device="cpu")

            payload["preprocessing"]["numeric_scales"][0] = 1.0
            original_reference = payload["training"]["patience_reference"]
            payload["training"]["patience_reference"] = (
                1.0 if original_reference != 1.0 else 0.0
            )
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checkpoint selection and stopping"):
                FTTransformerAdapter.load(path, device="cpu")

            payload["training"]["patience_reference"] = original_reference
            first_key = next(iter(payload["state_dict"]))
            del payload["state_dict"][first_key]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "state does not match"):
                FTTransformerAdapter.load(path, device="cpu")

    def test_load_rejects_nonfinite_model_tensors(self):
        adapter, _ = self.fitted_adapter()
        torch = adapter._torch
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            first_key = next(iter(payload["state_dict"]))
            original = payload["state_dict"][first_key].clone()
            for value in (float("nan"), float("inf")):
                with self.subTest(value=value):
                    payload["state_dict"][first_key] = torch.full_like(original, value)
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "non-finite tensors"):
                        FTTransformerAdapter.load(path, device="cpu")
            payload["state_dict"][first_key] = original

    def test_artifact_contains_only_tensor_state_and_primitive_metadata(self):
        adapter, _ = self.fitted_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapter.artifact_name
            adapter.save(path)
            payload = adapter._torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format_version"], 1)
            self.assertEqual(payload["class_meaning"], {"positive_label": 1, "probability": "P(y=1)"})
            self.assertTrue(all(value.device.type == "cpu" for value in payload["state_dict"].values()))
            self.assertTrue(all(value.dtype == adapter._torch.float32 for value in payload["state_dict"].values()))
            self.assertRegex(payload["training"]["training_fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
