from __future__ import annotations

import math
from typing import Any


FT_DEFAULT_PARAMS: dict[str, Any] = {
    "n_blocks": 3,
    "d_block": 192,
    "attention_n_heads": 8,
    "attention_dropout": 0.2,
    "ffn_d_hidden": None,
    "ffn_d_hidden_multiplier": 4 / 3,
    "ffn_activation": "ReGLU",
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "train_batch_size": 256,
    "max_epochs": 200,
    "validation_frequency": 1,
    "validation_batch_size": 1024,
    "patience": 20,
    "min_delta": 0.001,
    "preprocessing": "standard_scaler",
    "unknown_category_policy": "training_mode",
    "precision": "float32",
    "device": "cuda:0",
    "class_weighting": "none",
}
FT_SUPPORTED_PARAMS = frozenset(FT_DEFAULT_PARAMS)


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def resolve_ft_transformer_params(params: object) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("model_params must be a dictionary.")
    if any(not isinstance(key, str) for key in params):
        raise ValueError("FT-Transformer model parameter names must be strings.")
    unknown = set(params) - FT_SUPPORTED_PARAMS
    if unknown:
        raise ValueError(f"Unsupported FT-Transformer parameters: {sorted(unknown)}")
    resolved = {**FT_DEFAULT_PARAMS, **params}
    validate_ft_transformer_params(resolved)
    return resolved


def validate_ft_transformer_params(params: dict[str, Any]) -> None:
    if set(params) != FT_SUPPORTED_PARAMS:
        missing = sorted(FT_SUPPORTED_PARAMS - set(params))
        unknown = sorted(set(params) - FT_SUPPORTED_PARAMS)
        raise ValueError(
            f"FT-Transformer parameters are incomplete or unsupported: "
            f"missing={missing}, unknown={unknown}"
        )

    integer_minimums = {
        "n_blocks": 1,
        "d_block": 1,
        "attention_n_heads": 1,
        "train_batch_size": 1,
        "max_epochs": 1,
        "validation_frequency": 1,
        "validation_batch_size": 1,
        "patience": 1,
    }
    for name, minimum in integer_minimums.items():
        value = params[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    if params["n_blocks"] > 6:
        raise ValueError("n_blocks must be between 1 and 6 for the installed default configurations.")
    if params["d_block"] % params["attention_n_heads"] != 0:
        raise ValueError("d_block must be divisible by attention_n_heads.")

    hidden = params["ffn_d_hidden"]
    multiplier = params["ffn_d_hidden_multiplier"]
    if hidden is not None and (type(hidden) is not int or hidden <= 0):
        raise ValueError("ffn_d_hidden must be a positive integer or None.")
    if multiplier is not None and (
        not _finite_number(multiplier) or multiplier <= 0
    ):
        raise ValueError("ffn_d_hidden_multiplier must be positive and finite or None.")
    if (hidden is None) == (multiplier is None):
        raise ValueError("Specify exactly one of ffn_d_hidden and ffn_d_hidden_multiplier.")

    for name in ("attention_dropout", "ffn_dropout", "residual_dropout"):
        value = params[name]
        if not _finite_number(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0, 1).")
    for name in ("learning_rate", "weight_decay", "min_delta"):
        value = params[name]
        if not _finite_number(value):
            raise ValueError(f"{name} must be finite.")
        if value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(
                f"{name} must be {'positive' if name == 'learning_rate' else 'nonnegative'}."
            )
    if params["ffn_activation"] not in ("ReGLU", "ReLU"):
        raise ValueError("ffn_activation must be 'ReGLU' or 'ReLU'.")
    if params["preprocessing"] != "standard_scaler":
        raise ValueError("preprocessing must be 'standard_scaler'.")
    if params["unknown_category_policy"] != "training_mode":
        raise ValueError("unknown_category_policy must be 'training_mode'.")
    if params["precision"] != "float32":
        raise ValueError("Only float32 precision is supported.")
    if params["class_weighting"] != "none":
        raise ValueError("class_weighting must be 'none'.")
    if params["validation_frequency"] != 1:
        raise ValueError("validation_frequency must equal 1.")
    device = params["device"]
    if not isinstance(device, str) or not (
        device == "cpu"
        or (device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit())
    ):
        raise ValueError("device must be explicit: 'cpu' or 'cuda:<index>'.")
