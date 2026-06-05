#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-${HOME}/pyjuliapkg_pysr_acceptance}"

log() {
  printf '[acceptance-deploy] %s\n' "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '缺少命令: %s\n' "$1" >&2
    exit 1
  fi
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

create_env_if_missing() {
  local env_name="$1"
  local python_version="$2"
  shift 2
  if conda_env_exists "${env_name}"; then
    log "环境 ${env_name} 已存在，跳过 conda create"
    return
  fi
  log "创建环境 ${env_name}"
  conda create -y -n "${env_name}" "python=${python_version}" pip "$@" -c conda-forge
}

need_cmd conda

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  printf '请在仓库内运行本脚本，当前未找到 pyproject.toml: %s\n' "${REPO_ROOT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
log "仓库目录: ${REPO_ROOT}"
log "Julia/PySR 缓存目录: ${PYTHON_JULIAPKG_PROJECT}"

create_env_if_missing sim_base 3.10
log "安装 sim_base / PySR 依赖"
conda run -n sim_base python -m pip install -U pip
conda run -n sim_base python -m pip install -e .
conda run -n sim_base python -m pip install numpy pandas scipy scikit-learn sympy pyyaml gplearn pysr pyoperon

create_env_if_missing sim_llm 3.10 "sympy==1.13.1"
log "安装 sim_llm / LLMSR / DRSR 依赖"
conda run -n sim_llm python -m pip install -U pip
conda run -n sim_llm python -m pip install -e .
conda run -n sim_llm python -m pip install -r ./scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/requirements.txt
conda run -n sim_llm python -m pip install pyyaml scikit-learn

log "预热 PySR 导入"
PYSR_PREHEAT_PY="$(mktemp "${TMPDIR:-/tmp}/sim_acceptance_pysr_preheat.XXXXXX.py")"
cat > "${PYSR_PREHEAT_PY}" <<'PY'
from pysr import PySRRegressor
print("pysr_import_ok")
PY
PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT}" PYTHONPATH=. \
  conda run -n sim_base python "${PYSR_PREHEAT_PY}"
rm -f "${PYSR_PREHEAT_PY}"

log "基础导入检查"
SIM_IMPORT_PY="$(mktemp "${TMPDIR:-/tmp}/sim_acceptance_import.XXXXXX.py")"
cat > "${SIM_IMPORT_PY}" <<'PY'
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor
print("sim_import_ok")
PY
PYTHONPATH=. conda run -n sim_llm python "${SIM_IMPORT_PY}"
rm -f "${SIM_IMPORT_PY}"

log "部署完成"
log "下一步："
log "  bash acceptance/测试pysr.sh"
log "  SIM_LLM_CONFIG=/path/to/llm.config bash acceptance/测试llmsr.sh"
log "  SIM_LLM_CONFIG=/path/to/llm.config bash acceptance/测试drsr.sh"
