import importlib.util

import click.testing
import pytest

from aimnet.calculators import model_registry
from aimnet.cli import cli


def test_cli_help_smoke():
    result = click.testing.CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "export" in result.output


def test_train_help_smoke_without_eager_train_imports():
    result = click.testing.CliRunner().invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--no-default-config" in result.output


def test_calculators_star_import_without_optional_deps():
    from aimnet import calculators

    assert "AIMNet2Calculator" in calculators.__all__
    if importlib.util.find_spec("ase") is None:
        assert "AIMNet2ASE" not in calculators.__all__
    if importlib.util.find_spec("pysisyphus") is None:
        assert "AIMNet2Pysis" not in calculators.__all__


@pytest.fixture
def runner():
    return click.testing.CliRunner()


def test_download_requires_arguments(runner):
    result = runner.invoke(cli, ["download"])
    assert result.exit_code != 0
    assert "Specify model names or --all" in result.output


def test_download_unknown_model_fails(runner):
    result = runner.invoke(cli, ["download", "no-such-model"])
    assert result.exit_code != 0
    assert "not found in the registry" in str(result.output) + str(result.exception)


def test_download_named_model_fetches_it(runner, monkeypatch):
    fetched = []
    monkeypatch.setattr(model_registry, "get_registry_model_path", lambda name: fetched.append(name) or f"mock://{name}.pt")
    result = runner.invoke(cli, ["download", "aimnet2"])
    assert result.exit_code == 0, result.output
    assert fetched == [model_registry.resolve_registry_model_name("aimnet2")]


def test_download_all_fetches_every_registry_model(runner, monkeypatch):
    fetched = []
    monkeypatch.setattr(model_registry, "get_registry_model_path", lambda name: fetched.append(name) or f"mock://{name}.pt")
    result = runner.invoke(cli, ["download", "--all"])
    assert result.exit_code == 0, result.output
    assert sorted(fetched) == sorted(model_registry.load_model_registry()["models"])


def test_info_reports_environment(runner):
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0, result.output
    assert "aimnet" in result.output
    assert "torch" in result.output
    assert "warp-lang" in result.output
    assert "aimnet::conv_sv_2d_sp_fwd" in result.output
    assert "model cache" in result.output
