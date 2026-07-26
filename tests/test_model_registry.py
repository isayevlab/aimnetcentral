import hashlib
import re
from unittest.mock import Mock
from urllib.parse import urlparse

import pytest
import requests
import yaml

from aimnet.calculators.model_registry import (
    FamilyPolicy,
    get_cache_dir,
    get_family_policy,
    get_registry_model_family,
    load_model_registry,
    resolve_registry_model_name,
)


def test_load_model_registry_respects_registry_file_param(tmp_path):
    """registry_file param should override the default path."""
    fake = {"aliases": {}, "models": {"fake_model": {"file": "x.pt", "url": "http://x"}}}
    registry_path = tmp_path / "my_registry.yaml"
    registry_path.write_text(yaml.dump(fake))

    result = load_model_registry(str(registry_path))
    assert "fake_model" in result["models"]


def test_load_model_registry_uses_default_when_no_param():
    """When called with no args, loads the default registry with known models."""
    result = load_model_registry()
    assert "models" in result
    assert "aimnet2" in result.get("aliases", {})


def test_default_aimnet2_alias_resolves_to_wb97m_d3():
    """The bare `aimnet2` alias must resolve to the canonical wb97m-d3 member 0."""
    registry = load_model_registry()
    assert registry["aliases"]["aimnet2"] == "aimnet2-wb97m-d3_0"
    assert "aimnet2-wb97m-d3_0" in registry["models"]
    assert resolve_registry_model_name("aimnet2") == "aimnet2-wb97m-d3_0"
    assert get_registry_model_family("aimnet2") == "wb97m-d3"


def test_short_alias_forms_match():
    """Each model family's short alias forms (dash-canonical plus any legacy
    forms that have shipped publicly) must all resolve to the same canonical
    member-0 model key."""
    registry = load_model_registry()
    aliases = registry["aliases"]

    # (canonical_alias, [legacy_aliases], expected_target)
    expectations = [
        ("aimnet2-nse", ["aimnet2nse"], "aimnet2-nse_0"),
        ("aimnet2-pd", ["aimnet2pd"], "aimnet2-pd_0"),
        ("aimnet2-rxn", ["aimnet2rxn"], "aimnet2-rxn_0"),
        ("aimnet2-wb97m", ["aimnet2_wb97m"], "aimnet2-wb97m-d3_0"),
        ("aimnet2-b973c", ["aimnet2_b973c"], "aimnet2-b973c-d3_0"),
        ("aimnet2-2025", ["aimnet2_2025"], "aimnet2-b973c-2025-d3_0"),
    ]
    for canonical_alias, legacy_aliases, expected_target in expectations:
        assert aliases.get(canonical_alias) == expected_target, f"{canonical_alias} should resolve to {expected_target}"
        for legacy in legacy_aliases:
            assert aliases.get(legacy) == expected_target, f"legacy alias {legacy} should resolve to {expected_target}"


def test_registered_model_family_inference_for_short_aliases():
    """Family tags used for energy-scale warnings must follow the canonical registry key."""
    expectations = {
        "aimnet2": "wb97m-d3",
        "aimnet2-wb97m": "wb97m-d3",
        "aimnet2-b973c": "b973c-d3",
        "aimnet2-2025": "b973c-2025-d3",
        "aimnet2-nse": "nse",
        "aimnet2-pd": "pd",
        "aimnet2-rxn": "rxn",
        "aimnet2rxn": "rxn",
    }
    for alias, family in expectations.items():
        assert get_registry_model_family(alias) == family


def test_canonical_keys_for_all_families():
    """Every model family must have its four ensemble members registered under
    the canonical dash-form key, mapped to the original (unchanged) GCS files."""
    registry = load_model_registry()
    models = registry["models"]

    expected = [
        # (canonical_key_template, file_template, gcs_subdir)
        ("aimnet2-wb97m-d3_{i}", "aimnet2_wb97m_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-b973c-d3_{i}", "aimnet2_b973c_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-b973c-2025-d3_{i}", "aimnet2_2025_b973c_d3_{i}.pt", "AIMNet2"),
        ("aimnet2-nse_{i}", "aimnet2nse_wb97m_{i}.pt", "AIMNet2NSE"),
        ("aimnet2-pd_{i}", "aimnet2-pd_{i}.pt", "AIMNet2Pd"),
        ("aimnet2-rxn_{i}", "aimnet2_rxn_{i}.pt", "AIMNet2rxn"),
    ]
    base = "https://storage.googleapis.com/aimnetcentral/aimnet2v2"
    for key_tmpl, file_tmpl, subdir in expected:
        for i in range(4):
            key = key_tmpl.format(i=i)
            file = file_tmpl.format(i=i)
            assert key in models, f"missing model key: {key}"
            entry = models[key]
            assert entry["file"] == file, f"{key}: file mismatch"
            assert entry["url"] == f"{base}/{subdir}/{file}", f"{key}: url mismatch"


def test_legacy_member_aliases_resolve_via_loader(monkeypatch):
    """End-to-end: every legacy member-level key must resolve through
    get_registry_model_path's alias indirection to the canonical model's file.
    The download step is stubbed so the test exercises the lookup logic only."""
    from aimnet.calculators import model_registry as mr

    # stub the download step so the loader collapses to pure lookup; return
    # the expected on-disk path get_registry_model_path would normally hand back
    monkeypatch.setattr(
        mr,
        "_maybe_download_asset",
        lambda file, url, expected_sha256: f"/assets/{file}",
    )

    registry = mr.load_model_registry()
    legacy_keys = [
        # underscore-form legacy keys (the previous shape of every default model)
        "aimnet2_wb97m_d3_0",
        "aimnet2_wb97m_d3_1",
        "aimnet2_wb97m_d3_2",
        "aimnet2_wb97m_d3_3",
        "aimnet2_b973c_d3_0",
        "aimnet2_b973c_d3_1",
        "aimnet2_b973c_d3_2",
        "aimnet2_b973c_d3_3",
        "aimnet2_b973c_2025_d3_0",
        "aimnet2_b973c_2025_d3_1",
        "aimnet2_b973c_2025_d3_2",
        "aimnet2_b973c_2025_d3_3",
        "aimnet2_rxn_0",
        "aimnet2_rxn_1",
        "aimnet2_rxn_2",
        "aimnet2_rxn_3",
        # no-separator-form legacy keys
        "aimnet2nse_0",
        "aimnet2nse_1",
        "aimnet2nse_2",
        "aimnet2nse_3",
    ]
    for legacy in legacy_keys:
        path = mr.get_registry_model_path(legacy)
        canonical = registry["aliases"][legacy]
        expected_file = registry["models"][canonical]["file"]
        assert path == f"/assets/{expected_file}", f"{legacy} resolved to {path}, expected /assets/{expected_file}"


def test_no_alias_to_alias_chains():
    """Single-hop invariant: every alias value must be a real model key, never
    another alias. This makes the get_registry_model_path one-hop lookup
    mechanically enforced rather than maintained by hand."""
    registry = load_model_registry()
    models = registry["models"]
    aliases = registry["aliases"]

    for src, dst in aliases.items():
        assert dst in models, f"alias {src!r} -> {dst!r} is not a model entry"
        assert dst not in aliases, f"alias {src!r} -> {dst!r} is itself an alias (would require >1 hop)"


def test_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(tmp_path))
    assert get_cache_dir() == str(tmp_path)


def test_registry_sha256_entries_are_valid_hex():
    registry = load_model_registry()
    for key, entry in registry["models"].items():
        digest = entry.get("sha256")
        if digest is None:
            continue
        assert len(digest) == 64, key
        int(digest, 16)


def test_every_registry_model_has_sha256():
    registry = load_model_registry()
    assert len(registry["models"]) == 24
    for key, entry in registry["models"].items():
        digest = entry.get("sha256")
        assert isinstance(digest, str), key
        assert re.fullmatch(r"[0-9a-f]{64}", digest), key


@pytest.mark.parametrize("digest", [None, "", 12, "A" * 64, "0" * 63, "g" * 64])
def test_get_registry_model_path_rejects_missing_or_invalid_sha256(monkeypatch, tmp_path, digest):
    from aimnet.calculators import model_registry as mr

    registry = {
        "aliases": {},
        "models": {"custom": {"file": "custom.pt", "url": "https://example.invalid/custom.pt", "sha256": digest}},
    }
    monkeypatch.setattr(mr, "load_model_registry", lambda registry_file=None: registry)
    download = Mock(side_effect=AssertionError("download must not be attempted"))
    monkeypatch.setattr(mr, "_maybe_download_asset", download)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(cache_dir))

    with pytest.raises(ValueError, match="SHA-256"):
        mr.get_registry_model_path("custom")

    download.assert_not_called()
    assert not cache_dir.exists()


def test_cache_hit_revalidates_sha256(monkeypatch, tmp_path):
    from aimnet.calculators import model_registry as mr

    content = b"corrupt cache"
    expected = "0" * 64
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "custom.pt").write_bytes(content)
    registry = {
        "aliases": {},
        "models": {"custom": {"file": "custom.pt", "url": "https://example.invalid/custom.pt", "sha256": expected}},
    }
    monkeypatch.setattr(mr, "load_model_registry", lambda registry_file=None: registry)
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(cache_dir))

    with pytest.raises(ValueError, match=f"expected {expected}"):
        mr.get_registry_model_path("custom")

    assert (cache_dir / "custom.pt").read_bytes() == content


def test_atomic_download_rejects_bad_sha256(monkeypatch, tmp_path):
    from aimnet.calculators import model_registry as mr

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.iter_content.return_value = [b"downloaded bytes"]
    monkeypatch.setattr(mr.requests, "get", Mock(return_value=response))
    target = tmp_path / "custom.pt"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        mr._download_asset_atomic(str(target), "https://example.invalid/custom.pt", "0" * 64)

    assert not target.exists()
    assert not list(tmp_path.glob(".download-*.tmp"))


def test_atomic_download_installs_verified_bytes(monkeypatch, tmp_path):
    from aimnet.calculators import model_registry as mr

    content = b"verified bytes"
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.iter_content.return_value = [content]
    monkeypatch.setattr(mr.requests, "get", Mock(return_value=response))
    target = tmp_path / "custom.pt"

    mr._download_asset_atomic(str(target), "https://example.invalid/custom.pt", hashlib.sha256(content).hexdigest())

    assert target.read_bytes() == content


def test_registry_alias_takes_precedence_over_implicit_local_path(monkeypatch, tmp_path):
    from aimnet.calculators import model_registry as mr

    content = b"registered artifact"
    digest = hashlib.sha256(content).hexdigest()
    registry = {
        "aliases": {"alias": "canonical"},
        "models": {
            "canonical": {
                "family": "custom",
                "file": "alias",
                "url": "https://example.invalid/alias",
                "sha256": digest,
            }
        },
    }
    monkeypatch.setattr(mr, "load_model_registry", lambda registry_file=None: registry)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(cache_dir))
    downloaded = tmp_path / "downloaded"
    downloaded.write_bytes(content)
    monkeypatch.setattr(mr, "_maybe_download_asset", lambda **kwargs: str(downloaded))

    implicit = tmp_path / "alias"
    implicit.write_bytes(b"shadowing local file")
    monkeypatch.chdir(tmp_path)
    assert mr.get_model_path("alias") == str(downloaded)
    assert mr.get_model_path("./alias") == "./alias"


@pytest.mark.network
def test_registry_digests_match():
    registry = load_model_registry()
    for name, entry in registry["models"].items():
        digest = hashlib.sha256()
        with requests.get(entry["url"], stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            for item in [*response.history, response]:
                url = urlparse(item.url)
                assert (
                    url.scheme == "https"
                    and url.hostname == "storage.googleapis.com"
                    and url.path.startswith("/aimnetcentral/")
                ), f"{name} redirected to unexpected origin: {item.url}"
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"], f"{name}: {entry['url']}"


def test_every_yaml_family_resolves_to_a_policy():
    """Every family block in the registry YAML must resolve via get_family_policy
    to a non-neutral policy that carries its own tag and at least one member."""
    registry = load_model_registry()
    for family in registry["families"]:
        policy = get_family_policy(family)
        assert policy.family == family
        assert policy.members, f"family {family!r} has no registry members"


def test_unknown_family_returns_neutral_policy():
    """Unknown or undeclared families must get the neutral policy, not raise —
    raw nn.Module loads and third-party checkpoints rely on this."""
    for family in ("no-such-family", None):
        policy = get_family_policy(family)
        assert policy == FamilyPolicy()
        assert policy.family is None
        assert policy.supports_charged_systems is None
        assert policy.posthoc_d3_params is None
        assert policy.members == ()


def test_every_registry_model_has_a_family_policy_block():
    """Every model entry must declare a family that names a block under `families:`,
    and get_registry_model_family must return exactly that declared tag."""
    registry = load_model_registry()
    for key, entry in registry["models"].items():
        family = entry.get("family")
        assert family is not None, f"model {key!r} does not declare a family"
        assert family in registry["families"], f"model {key!r} family {family!r} has no policy block"
        assert get_registry_model_family(key) == family


def test_family_policy_members_are_in_ensemble_order():
    """FamilyPolicy.members must list the family's registry keys in member order,
    so ensemble_member indices map onto the correct checkpoints."""
    registry = load_model_registry()
    for family in registry["families"]:
        members = get_family_policy(family).members
        expected = tuple(k for k, v in registry["models"].items() if v.get("family") == family)
        assert members == expected
        for i, member in enumerate(members):
            assert member.endswith(f"_{i}"), f"{member} is not ensemble member {i}"


def test_rxn_family_policy_pins_posthoc_wb97m_d3():
    """The rxn family policy must carry the AIMNet2 wB97M D3(BJ) parameters and the
    charged-systems restriction previously hardcoded in the calculator."""
    policy = get_family_policy("rxn")
    assert policy.supports_charged_systems is False
    assert policy.posthoc_d3_params == {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}


def test_non_rxn_family_policies_are_permissive():
    """All non-rxn released families carry their policy inside the .pt metadata:
    no charge restriction and no post-hoc dispersion defaults."""
    registry = load_model_registry()
    for family in registry["families"]:
        if family == "rxn":
            continue
        policy = get_family_policy(family)
        assert policy.supports_charged_systems is None, family
        assert policy.posthoc_d3_params is None, family
