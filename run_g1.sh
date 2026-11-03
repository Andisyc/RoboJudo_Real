#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ROBOJUDO_PYTHON="${ROBOJUDO_PYTHON:-python}"
export UNITREE_NET_IF="${UNITREE_NET_IF:-enP8p1s0}"
CONFIG="${ROBOJUDO_CONFIG:-g1_native_loco_mimic_23dof_backpack}"

case "${1:-}" in
    ""|--check-only) ;;
    *) echo "Usage: ./run_g1.sh [--check-only]" >&2; exit 2 ;;
esac

# This script reads configuration only; it cannot load a hardware bridge.
env -u LD_PRELOAD -u LD_LIBRARY_PATH -u CYCLONEDDS_URI \
    "$ROBOJUDO_PYTHON" scripts/check_g1_native_config.py \
    -c "$CONFIG" --net-if "$UNITREE_NET_IF"

if [[ "${1:-}" == "--check-only" ]]; then
    echo "检查完成，没有启动机器人。"
    exit 0
fi

if pgrep -f '[p]ython.*scripts/run_pipeline.py' >/dev/null; then
    echo "错误：已有 RoboJudo 控制程序运行。" >&2
    exit 1
fi

echo "启动配置：$CONFIG；舞蹈中 A/Select 会等待站姿，不是急停。"
exec env -u LD_PRELOAD -u LD_LIBRARY_PATH -u CYCLONEDDS_URI \
    bash ./start.sh -c "$CONFIG"
