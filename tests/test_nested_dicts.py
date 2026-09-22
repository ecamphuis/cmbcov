"""
``missing`` key policy and shape/type errors for the nested-dict operators.

``_operate_nested_dicts`` (via ``add_nested_dicts``/``mult_nested_dicts``)
takes an explicit ``missing`` policy per call rather than silently carrying
a one-sided key through: that matters for ``spectra.py``'s ``data_model``
multiplications, where silently carrying a key through would hide
beams/pixwin/fl not being applied (it is fine for ``generator.py``'s
``nl: {}`` case). A shape/type mismatch on a leaf raises, naming the full
nested key path, rather than silently keeping the first operand.
"""

import numpy as np
import pytest

from cmbcov.utils.array_utils import (
    add_nested_dicts,
    mult_nested_dicts,
)


@pytest.mark.parametrize("op", [add_nested_dicts, mult_nested_dicts])
def test_missing_key_error_default(op):
    d1 = {"a": {"x": 1}}
    d2 = {"a": {"x": 1}, "b": {"y": 2}}
    with pytest.raises(KeyError, match=r"\['b'\]"):
        op(d1, d2)


@pytest.mark.parametrize("op", [add_nested_dicts, mult_nested_dicts])
def test_missing_key_error_explicit_names_side(op):
    d1 = {"a": {"x": 1}, "b": {"y": 2}}
    d2 = {"a": {"x": 1}}
    with pytest.raises(KeyError, match=r"\['b'\].*dict2"):
        op(d1, d2, missing="error")


def test_add_missing_key_identity_is_zero():
    d1 = {"a": {"x": 1}, "b": {"y": 2}}
    d2 = {"a": {"x": 10}}
    result = add_nested_dicts(d1, d2, missing="identity")
    assert result == {"a": {"x": 11}, "b": {"y": 2}}


def test_mult_missing_key_identity_is_one():
    d1 = {"a": {"x": 3}, "b": {"y": 2}}
    d2 = {"a": {"x": 10}}
    result = mult_nested_dicts(d1, d2, missing="identity")
    assert result == {"a": {"x": 30}, "b": {"y": 2}}


def test_missing_key_identity_either_side():
    # Key missing from dict1 rather than dict2.
    d1 = {"a": {"x": 1}}
    d2 = {"a": {"x": 1}, "b": {"y": 2}}
    result = add_nested_dicts(d1, d2, missing="identity")
    assert result == {"a": {"x": 2}, "b": {"y": 2}}


@pytest.mark.parametrize("op", [add_nested_dicts, mult_nested_dicts])
def test_length_mismatch_raises_with_path(op):
    d1 = {"090GHz150GHz": {"TE": np.ones(701)}}
    d2 = {"090GHz150GHz": {"TE": np.ones(600)}}
    with pytest.raises(
        ValueError, match=r"\['090GHz150GHz'\]\['TE'\].*\(701,\).*\(600,\)"
    ):
        op(d1, d2, missing="error")


@pytest.mark.parametrize("op", [add_nested_dicts, mult_nested_dicts])
def test_dict_vs_non_dict_raises_with_path(op):
    d1 = {"090GHz150GHz": {"TE": {"nested": 1}}}
    d2 = {"090GHz150GHz": {"TE": np.ones(5)}}
    with pytest.raises(ValueError, match=r"\['090GHz150GHz'\]\['TE'\]"):
        op(d1, d2, missing="error")


def test_bad_missing_policy_raises():
    with pytest.raises(ValueError):
        add_nested_dicts({"a": 1}, {"a": 1}, missing="bogus")


def test_no_missing_keys_both_policies_agree():
    d1 = {"a": {"x": 1, "y": 2}, "b": 3}
    d2 = {"a": {"x": 10, "y": 20}, "b": 30}
    assert add_nested_dicts(d1, d2, missing="error") == add_nested_dicts(
        d1, d2, missing="identity"
    )
    assert mult_nested_dicts(d1, d2, missing="error") == mult_nested_dicts(
        d1, d2, missing="identity"
    )
