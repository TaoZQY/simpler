#!/usr/bin/env bash

# Configure the Simpler build/runtime environment for 910B or 910C, then
# rebuild the editable Python package. Source this script when you want the
# exported variables and virtualenv activation to remain in the current shell.

_SETUP_910_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SETUP_910_PROJECT_ROOT="$(cd "${_SETUP_910_SCRIPT_DIR}/.." && pwd)"
_SETUP_910_REPO_ROOT="$(cd "${_SETUP_910_PROJECT_ROOT}/.." && pwd)"
_SETUP_910_WORKSPACE_ROOT="$(cd "${_SETUP_910_REPO_ROOT}/.." && pwd)"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _SETUP_910_SOURCED=0
else
  _SETUP_910_SOURCED=1
fi

_setup_910_usage() {
  cat <<'USAGE'
setup_910_env_and_build.sh - setup 910B/910C env, activate venv, set block_dim, and build

Usage:
  source ./tools/setup_910_env_and_build.sh
  source ./tools/setup_910_env_and_build.sh --chip 910b
  source ./tools/setup_910_env_and_build.sh --chip 910c
  source ./tools/setup_910_env_and_build.sh --block-dim 20
  ./tools/setup_910_env_and_build.sh --chip 910b

Options:
  --chip 910b|910c       Environment preset. Default: 910b
  --block-dim N          910B only: export SIMPLER_BLOCK_DIM. Default: 20
  --no-build             910B only: setup environment; skip pip editable build
  -h, --help             Show this help

Environment overrides:
  CANN_HOME              910B CANN root. Default: /home/zt/Ascend/cann-8.5.0
  PTO_ISA_ROOT           910B PTO ISA root. Default: <simpler>/build/pto-isa when present,
                         otherwise $CANN_HOME/pto-isa
  VENV_DIR               Python venv. Default: <simpler>/.venv
  SET_PTO_ENV            910C set_pto_env.sh path. Default search locations:
                          /home/x00611368/set_pto_env.sh
                          <workspace>/set_pto_env.sh
                          <HW-PTO_RTS>/set_pto_env.sh
                          <simpler>/set_pto_env.sh

Notes:
  910B sources $CANN_HOME/set_env.sh, activates venv, exports block_dim, and builds.
  910C only sources set_pto_env.sh, then returns without venv activation or build.
  Prefer "source" if you want the env to remain active after the script exits.
USAGE
}

_setup_910_find_set_pto_env() {
  local explicit_path="$1"
  local candidates=()
  local candidate

  if [[ -n "${explicit_path}" ]]; then
    candidates+=("${explicit_path}")
  fi

  candidates+=(
    "/home/x00611368/set_pto_env.sh"
    "${_SETUP_910_WORKSPACE_ROOT}/set_pto_env.sh"
    "${_SETUP_910_REPO_ROOT}/set_pto_env.sh"
    "${_SETUP_910_PROJECT_ROOT}/set_pto_env.sh"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

_setup_910_default_pto_isa_root() {
  local cann_home="$1"
  local local_pto_isa="${_SETUP_910_PROJECT_ROOT}/build/pto-isa"

  if [[ -d "${local_pto_isa}" ]]; then
    printf '%s\n' "${local_pto_isa}"
  else
    printf '%s\n' "${cann_home}/pto-isa"
  fi
}

_setup_910_main() {
  local chip="${SIMPLER_CHIP:-910b}"
  local block_dim="${SIMPLER_BLOCK_DIM:-20}"
  local do_build=1
  local cann_home="${CANN_HOME:-/home/zt/Ascend/cann-8.5.0}"
  local venv_dir="${VENV_DIR:-${_SETUP_910_PROJECT_ROOT}/.venv}"
  local set_pto_env="${SET_PTO_ENV:-}"
  local old_pwd
  local rc

  while (($# > 0)); do
    case "$1" in
      --chip)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --chip requires 910b or 910c." >&2
          return 2
        fi
        chip="$2"
        shift 2
        ;;
      --chip=*)
        chip="${1#*=}"
        shift
        ;;
      --block-dim)
        if [[ $# -lt 2 ]]; then
          echo "ERROR: --block-dim requires a positive integer." >&2
          return 2
        fi
        block_dim="$2"
        shift 2
        ;;
      --block-dim=*)
        block_dim="${1#*=}"
        shift
        ;;
      --no-build)
        do_build=0
        shift
        ;;
      -h|--help)
        _setup_910_usage
        return 0
        ;;
      *)
        echo "ERROR: unknown argument '$1'." >&2
        _setup_910_usage >&2
        return 2
        ;;
    esac
  done

  case "${chip}" in
    910b|910B)
      chip="910b"
      ;;
    910c|910C)
      chip="910c"
      ;;
    *)
      echo "ERROR: --chip must be 910b or 910c, got '${chip}'." >&2
      return 2
      ;;
  esac

  if ! [[ "${block_dim}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --block-dim must be a positive integer, got '${block_dim}'." >&2
    return 2
  fi

  if [[ "${_SETUP_910_SOURCED}" != "1" ]]; then
    echo "INFO: script is running in a child shell; source it if you need the env to persist."
  fi

  if [[ "${chip}" == "910b" ]]; then
    if [[ ! -f "${cann_home}/set_env.sh" ]]; then
      echo "ERROR: cannot find 910B CANN env file: ${cann_home}/set_env.sh" >&2
      return 1
    fi
    if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
      export PTO_ISA_ROOT="$(_setup_910_default_pto_isa_root "${cann_home}")"
    fi
    if [[ ! -d "${PTO_ISA_ROOT}" ]]; then
      echo "ERROR: PTO_ISA_ROOT does not exist: ${PTO_ISA_ROOT}" >&2
      return 1
    fi

    echo "INFO: sourcing 910B CANN env: ${cann_home}/set_env.sh"
    # shellcheck source=/dev/null
    source "${cann_home}/set_env.sh"
    export PTO_ISA_ROOT
  else
    set_pto_env="$(_setup_910_find_set_pto_env "${set_pto_env}")"
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
      echo "ERROR: cannot find set_pto_env.sh for 910C." >&2
      echo "ERROR: pass SET_PTO_ENV=/path/to/set_pto_env.sh or place it next to HW-PTO_RTS." >&2
      return 1
    fi

    echo "INFO: sourcing 910C PTO env: ${set_pto_env}"
    # shellcheck source=/dev/null
    source "${set_pto_env}"

    echo "INFO: 910C environment ready from ${set_pto_env}"
    return 0
  fi

  if [[ ! -f "${venv_dir}/bin/activate" ]]; then
    echo "ERROR: cannot find Python virtualenv: ${venv_dir}/bin/activate" >&2
    return 1
  fi

  echo "INFO: activating Python venv: ${venv_dir}"
  # shellcheck source=/dev/null
  source "${venv_dir}/bin/activate"

  export SIMPLER_BLOCK_DIM="${block_dim}"
  unset SIMPLER_PIPELINE_STRATEGY
  unset SIMPLER_AICPU_THREAD_NUM

  old_pwd="${PWD}"
  cd "${_SETUP_910_PROJECT_ROOT}" || return 1

  if [[ "${do_build}" == "1" ]]; then
    echo "INFO: rebuilding editable package in ${_SETUP_910_PROJECT_ROOT}"
    python -m pip install --no-build-isolation -e '.[test]'
    rc=$?
  else
    echo "INFO: skipping build because --no-build was passed."
    rc=0
  fi

  cd "${old_pwd}" || return 1

  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: build failed with exit code ${rc}." >&2
    return "${rc}"
  fi

  echo "INFO: environment ready: chip=${chip}, SIMPLER_BLOCK_DIM=${SIMPLER_BLOCK_DIM}"
  if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
    echo "INFO: PTO_ISA_ROOT=${PTO_ISA_ROOT}"
  fi
  echo "INFO: next benchmark example:"
  echo "INFO:   ./tools/benchmark_rounds.sh -p a2a3 -d 0 -r tensormap_and_ringbuffer -n 100 -v"

  return 0
}

_setup_910_main "$@"
_SETUP_910_RC=$?

unset -f _setup_910_usage
unset -f _setup_910_find_set_pto_env
unset -f _setup_910_default_pto_isa_root
unset -f _setup_910_main
unset _SETUP_910_SCRIPT_DIR
unset _SETUP_910_PROJECT_ROOT
unset _SETUP_910_REPO_ROOT
unset _SETUP_910_WORKSPACE_ROOT
unset _SETUP_910_SOURCED

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exit "${_SETUP_910_RC}"
else
  return "${_SETUP_910_RC}"
fi
