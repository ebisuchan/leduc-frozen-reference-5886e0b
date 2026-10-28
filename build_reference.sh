#!/usr/bin/env bash
# Public upstream only. A timeout/failure ends the run; no retry or wheel fallback.
set -euo pipefail
frozen_sha=5886e0b822bcc469cacb53a82c18ac1e8748d820
task_dir="$RUNNER_TEMP/frozen-leduc"
mkdir -p "$task_dir"
git clone --no-checkout https://github.com/google-deepmind/open_spiel.git "$task_dir/upstream"
git -C "$task_dir/upstream" checkout --detach "$frozen_sha"
test "$(git -C "$task_dir/upstream" rev-parse HEAD)" = "$frozen_sha"
export DEFAULT_OPTIONAL_DEPENDENCY=OFF
export OPEN_SPIEL_ENABLE_JAX=OFF OPEN_SPIEL_ENABLE_PYTORCH=OFF
export OPEN_SPIEL_ENABLE_PYTHON_MISC=OFF OPEN_SPIEL_BUILD_WITH_PYTHON=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF
export OPEN_SPIEL_BUILD_WITH_GAMUT=OFF OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF
export OS_PYTHON_VERSION=3.12 CXX=clang++ CC=clang
cd "$task_dir/upstream"
# Follow frozen upstream dependency installer; record exact dependency commits below.
./install.sh python3
python3 -m venv "$task_dir/venv"
py="$task_dir/venv/bin/python"
"$py" -m pip install --disable-pip-version-check --only-binary=:all: setuptools wheel numpy
"$py" -m pip wheel --no-deps --no-build-isolation . --wheel-dir "$task_dir/wheels"
"$py" -m pip install --no-deps "$task_dir"/wheels/*.whl
test "$(git rev-parse HEAD)" = "$frozen_sha"
git diff --exit-code HEAD
printf 'FROZEN_COMMIT=%s\n' "$frozen_sha"
find "$task_dir/wheels" -type f -name '*.whl' -exec sha256sum {} \;
for dep in pybind11 open_spiel/abseil-cpp open_spiel/json open_spiel/pybind11_json open_spiel/pybind11_abseil open_spiel/games/bridge/double_dummy_solver; do
  printf 'DEPENDENCY=%s SHA=' "$dep"
  git -C "$dep" rev-parse HEAD
done
"$py" -m pip freeze
clang++ --version
cmake --version
cd "$task_dir"
timeout --signal=TERM --kill-after=5s 90s "$py" "$RUNNER_TEMP/export_reference.py" "$task_dir/upstream"
