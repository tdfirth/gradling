from dataclasses import is_dataclass

import pytest

from gradling.configs import Config, get_config, list_pipelines, pipeline


class ExplicitNameConfig(Config, name="test_explicit_config"):
    value: int = 1


class SnakeCaseFallbackConfig(Config):
    value: int = 1


class EntrypointConfig(Config, name="test_entrypoint_config"):
    value: int = 1


class MultiLevelBase(Config, name="test_multi_level_base"):
    base_value: int = 1


class MultiLevelMid(MultiLevelBase):
    mid_value: int = 2


class MultiLevelLeaf(MultiLevelMid, name="test_multi_level_leaf"):
    leaf_value: int = 3


class OverrideEntrypointBase(Config, name="test_override_entrypoint_base"):
    value: int = 1


class OverrideEntrypointMid(OverrideEntrypointBase):
    pass


class OverrideEntrypointLeaf(OverrideEntrypointMid):
    value: int = 2


@pipeline(EntrypointConfig, name="run")
def entrypoint_config_run(_: EntrypointConfig):
    return "ok"


@pipeline(MultiLevelBase, name="base_run")
def multi_level_base_run(_: MultiLevelBase):
    return "base"


@pipeline(MultiLevelLeaf, name="leaf_run")
def multi_level_leaf_run(_: MultiLevelLeaf):
    return "leaf"


@pipeline(OverrideEntrypointBase, name="run")
def override_entrypoint_base_run(_: OverrideEntrypointBase):
    return "base"


@pipeline(OverrideEntrypointLeaf, name="run")
def override_entrypoint_leaf_run(_: OverrideEntrypointLeaf):
    return "leaf"


def test_config_registration_with_explicit_name():
    assert get_config("test_explicit_config") is ExplicitNameConfig


def test_config_registration_uses_snake_case_by_default():
    assert get_config("snake_case_fallback_config") is SnakeCaseFallbackConfig


def test_subclasses_are_auto_dataclasses():
    assert is_dataclass(SnakeCaseFallbackConfig)


def test_pipeline_decorator_registers_function():
    assert "run" in list_pipelines(EntrypointConfig)


def test_multilevel_subclasses_are_all_registered():
    assert get_config("test_multi_level_base") is MultiLevelBase
    assert get_config("multi_level_mid") is MultiLevelMid
    assert get_config("test_multi_level_leaf") is MultiLevelLeaf


def test_multilevel_subclass_keeps_dataclass_fields():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.base_value == 10
    assert cfg.mid_value == 20
    assert cfg.leaf_value == 30


def test_multilevel_subclass_inherits_and_adds_pipelines():
    pipelines = list_pipelines(MultiLevelLeaf)
    assert "base_run" in pipelines
    assert "leaf_run" in pipelines


def test_child_instances_include_parent_defaults_and_fields():
    cfg = MultiLevelLeaf()
    assert cfg.base_value == 1
    assert cfg.mid_value == 2
    assert cfg.leaf_value == 3


def test_child_can_call_parent_pipeline_directly():
    cfg = MultiLevelLeaf()
    pipelines = list_pipelines(MultiLevelLeaf)
    assert pipelines["base_run"](cfg) == "base"
    assert pipelines["leaf_run"](cfg) == "leaf"


def test_multilevel_subclass_can_override_inherited_pipeline():
    cfg = OverrideEntrypointLeaf()
    pipelines = list_pipelines(OverrideEntrypointLeaf)
    assert pipelines["run"](cfg) == "leaf"


def test_duplicate_config_name_raises():
    class FirstConfig(Config, name="test_duplicate_name"):
        value: int = 1

    assert FirstConfig.config_name() == "test_duplicate_name"

    with pytest.raises(ValueError):

        class SecondConfig(Config, name="test_duplicate_name"):
            value: int = 1
