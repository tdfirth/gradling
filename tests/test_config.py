from dataclasses import dataclass

from gradling.config import Config, runtime_field


@dataclass
class MultiLevelBase(Config):
    base_value: int = 1


@dataclass
class MultiLevelMid(MultiLevelBase):
    mid_value: int = 2


@dataclass
class MultiLevelLeaf(MultiLevelMid):
    leaf_value: int = 3


def test_multilevel_subclass_keeps_parent_fields():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.base_value == 10
    assert cfg.mid_value == 20
    assert cfg.leaf_value == 30


def test_to_dict():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.to_dict() == dict(base_value=10, mid_value=20, leaf_value=30)


def test_from_dict():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.from_dict(dict(base_value=10, mid_value=20, leaf_value=30)) == cfg


def test_replace():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert cfg.replace(base_value=999) == MultiLevelLeaf(
        base_value=999, mid_value=20, leaf_value=30
    )


def test_fields():
    cfg = MultiLevelLeaf(base_value=10, mid_value=20, leaf_value=30)
    assert [f.name for f in cfg.fields] == ["base_value", "mid_value", "leaf_value"]


@dataclass
class RuntimeLeaf(MultiLevelMid):
    rt: str = runtime_field("")


def test_runtime_fields():
    cli_names = [f.name for f in RuntimeLeaf.cli_fields()]
    assert "rt" not in cli_names
    assert "base_value" in cli_names
