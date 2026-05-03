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
