#!/usr/bin/env bash
# Validate the torch / warp-lang / nvalchemiops coupling on a CUDA box across
# the supported PyTorch range. For each version: fresh venv -> resolver-coherent
# install -> `pytest -m gpu` -> deterministic energy/force dump. A same-run
# torch-2.9 baseline is then used to diff every other version.
#
# Usage:
#   bash scripts/gpu_validate.sh            # run the full matrix
#   DRY_RUN=1 bash scripts/gpu_validate.sh  # print the per-version commands only
#
# Tunables (env vars):
#   TORCH_VERSIONS  default "2.8 2.9 2.10 2.11 2.12"
#   CUDA_INDEX      default "https://download.pytorch.org/whl/cu126"
#   PYTHON          default "python3.12"
#   RESULTS         default "./gpu-validation-results"
#   BASELINE        default "2.9"
#   ENERGY_ATOL     default "1e-5"   (Hartree)
#   FORCE_ATOL      default "1e-4"   (Hartree/Angstrom)
set -u

TORCH_VERSIONS="${TORCH_VERSIONS:-2.8 2.9 2.10 2.11 2.12}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
PYTHON="${PYTHON:-python3.12}"
RESULTS="${RESULTS:-./gpu-validation-results}"
BASELINE="${BASELINE:-2.9}"
ENERGY_ATOL="${ENERGY_ATOL:-1e-5}"
FORCE_ATOL="${FORCE_ATOL:-1e-4}"
DRY_RUN="${DRY_RUN:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"
mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"  # absolutize so per-venv dumps and compare agree
STATUS="$RESULTS/status.json"

# Build status.json incrementally with python (always available via base env).
_set_status() {  # label key value
    "$PYTHON" - "$STATUS" "$1" "$2" "$3" <<'PY'
import json, sys
path, label, key, value = sys.argv[1:5]
try:
    d = json.load(open(path))
except FileNotFoundError:
    d = {}
d.setdefault(label, {})[key] = value
json.dump(d, open(path, "w"), indent=2)
PY
}

echo "repo=$REPO workdir=$WORKDIR results=$RESULTS cuda_index=$CUDA_INDEX python=$PYTHON"
echo "{}" > "$STATUS"

for V in $TORCH_VERSIONS; do
    echo "::: torch $V :::"
    VENV="$WORKDIR/$V"
    # `uv pip install` honors VIRTUAL_ENV; `uv run` does NOT, so execute the
    # suite/dump via the venv's own python directly. pytest is installed into
    # the venv explicitly (the [ase] extra does not pull the dev group).
    install_cmd="uv venv --python $PYTHON $VENV && \
        VIRTUAL_ENV=$VENV uv pip install 'torch==$V.*' --index-url $CUDA_INDEX && \
        VIRTUAL_ENV=$VENV uv pip install -e '$REPO[ase]' pytest"
    suite_cmd="'$VENV/bin/python' -m pytest '$REPO/tests' -m gpu"
    dump_cmd="'$VENV/bin/python' -m aimnet.validation.gpu_observables --out '$RESULTS/$V.json'"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  install: $install_cmd"
        echo "  suite:   $suite_cmd"
        echo "  dump:    $dump_cmd"
        continue
    fi

    if eval "$install_cmd"; then
        _set_status "$V" install ok
    else
        echo "  install FAILED for torch $V"
        _set_status "$V" install fail
        _set_status "$V" gpu_suite skipped
        continue
    fi

    if eval "$suite_cmd"; then _set_status "$V" gpu_suite ok; else _set_status "$V" gpu_suite fail; fi
    eval "$dump_cmd" || echo "  observables dump FAILED for torch $V"
done

if [ "$DRY_RUN" = "1" ]; then
    echo "(dry run; no compare)"
    exit 0
fi

echo "::: comparing against baseline $BASELINE :::"
# compare_observables is version-agnostic stdlib living in the aimnet package;
# run it from the repo's own environment (always has aimnet installed).
(cd "$REPO" && uv run --no-sync python -m aimnet.validation.compare_observables \
    "$RESULTS" --baseline "$BASELINE" --energy-atol "$ENERGY_ATOL" --force-atol "$FORCE_ATOL")
exit $?
