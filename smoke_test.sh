#!/usr/bin/env bash
# 肺结节分类工程一键冒烟测试入口。
# 默认优先使用仓库内 Windows 虚拟环境；也可通过 PYTHON_BIN 指定解释器。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x "${SCRIPT_DIR}/.venv/Scripts/python.exe" ]]; then
  python_bin="${SCRIPT_DIR}/.venv/Scripts/python.exe"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  python_bin="${SCRIPT_DIR}/.venv/bin/python"
else
  python_bin="python"
fi

exec "${python_bin}" "${SCRIPT_DIR}/smoke_test.py" "$@"
