import importlib.util

from click.testing import CliRunner

from aimnet.cli import cli


def test_cli_help_smoke():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "export" in result.output


def test_train_help_smoke_without_eager_train_imports():
    result = CliRunner().invoke(cli, ["train", "--help"])
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
