import jax
import pytest
from pydantic import ValidationError

from gradling.config import Config, _snake_case


def test_snake_case(subtests):
    cases = [
        ("Foo", "foo"),
        ("FooBar", "foo_bar"),
        ("FooBarBaz", "foo_bar_baz"),
        ("foobarbaz", "foobarbaz"),
        ("foo_bar_baz", "foo_bar_baz"),
    ]

    for input, want in cases:
        with subtests.test(f"Snakecase {input} -> {want}"):
            got = _snake_case(input)
            assert want == got


class ExplicitNameConfig(Config, name="test_explicit_config"):
    value: int = 1


class SnakeCaseFallbackConfig(Config):
    value: int = 1


class MultiLevelBase(Config):
    base_value: int = 1


class MultiLevelMid(MultiLevelBase):
    mid_value: int = 2


class MultiLevelLeaf(MultiLevelMid):
    leaf_value: int = 3


def test_config_name_with_explicit_name():
    assert ExplicitNameConfig.config_name() == "test_explicit_config"


def test_config_name_uses_snake_case_fallback():
    assert SnakeCaseFallbackConfig.config_name() == "snake_case_fallback_config"


def test_multilevel_subclass_keeps_parent_fields():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.base_value == 10
    assert cfg.mid_value == 20
    assert cfg.leaf_value == 30


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        MultiLevelLeaf.model_validate({"unknown_field": 1})


def test_subclass_and_grandchild_are_registered_as_pytrees():
    mid = MultiLevelMid(base_value=2, mid_value=3)
    leaf = MultiLevelLeaf(base_value=2, mid_value=3, leaf_value=4)

    assert jax.tree_util.tree_leaves(mid) == [2, 3]
    assert jax.tree_util.tree_leaves(leaf) == [2, 3, 4]

    mapped = jax.tree_util.tree_map(lambda x: x + 1, leaf)
    assert isinstance(mapped, MultiLevelLeaf)
    assert mapped.base_value == 3
    assert mapped.mid_value == 4
    assert mapped.leaf_value == 5
