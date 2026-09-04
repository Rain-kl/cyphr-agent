#!/bin/bash

# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

# 确保以脚本所在目录为基准路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PID_FILE="${SCRIPT_DIR}/agent.pid"
LOG_FILE="${SCRIPT_DIR}/agent.log"
ENV_FILE="${SCRIPT_DIR}/.env"
DOWNLOAD_PID_FILE="${SCRIPT_DIR}/download.pid"
DOWNLOAD_LOG_FILE="${SCRIPT_DIR}/download.log"
DOWNLOAD_INFO_FILE="${SCRIPT_DIR}/download.info.log"
MODELS_DIR="${SCRIPT_DIR}/models"

# 加载 .env 环境变量（如果存在）
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# 颜色输出
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

# 获取运行命令
get_run_cmd() {
    if command -v uv >/dev/null 2>&1; then
        echo "uv run python main.py"
    elif [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
        echo "${SCRIPT_DIR}/.venv/bin/python main.py"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3 main.py"
    else
        echo "python main.py"
    fi
}

# 检查 Agent 服务是否正在运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 检查模型下载任务是否正在运行
is_download_running() {
    if [ -f "$DOWNLOAD_PID_FILE" ]; then
        local pid
        pid=$(cat "$DOWNLOAD_PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 列出本地已下载的模型包
list_local_models() {
    if [ ! -d "$MODELS_DIR" ]; then
        echo -e "  ${YELLOW}(models/ 目录不存在)${RESET}"
        return
    fi

    local count=0
    for d in "$MODELS_DIR"/*; do
        if [ -d "$d" ]; then
            count=$((count + 1))
            local dirname
            dirname=$(basename "$d")
            local size
            size=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
            local status="${GREEN}[完整/就绪]${RESET}"
            if ls "$d"/*.part 1>/dev/null 2>&1 || ls "$d"/*.aria2 1>/dev/null 2>&1; then
                status="${YELLOW}[存在断点分块/未完成]${RESET}"
            elif [ ! -f "$d/config.json" ]; then
                status="${YELLOW}[缺少 config.json]${RESET}"
            fi
            echo -e "  - ${BOLD}${dirname}${RESET} (${size}) ${status}"
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}(暂无已下载模型)${RESET}"
    fi
}

# 启动服务
start_service() {
    echo -e "${CYAN}正在启动 Agent 服务...${RESET}"
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo -e "${YELLOW}Agent 服务已在运行中 (PID: ${pid})，无需重复启动。${RESET}"
        return 0
    fi

    # 清理残留的无效 PID 文件
    [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"

    local run_cmd
    run_cmd=$(get_run_cmd)

    # 后台启动进程并将输出追加重定向至日志文件
    nohup $run_cmd >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    # 等待 1.5 秒确认进程是否稳定存活
    sleep 1.5

    if kill -0 "$new_pid" 2>/dev/null; then
        echo -e "${GREEN}✓ Agent 服务启动成功！(PID: ${new_pid})${RESET}"
        echo -e "  日志输出路径: ${LOG_FILE}"
    else
        echo -e "${RED}✗ Agent 服务启动失败，进程未正常运行。${RESET}"
        echo -e "${YELLOW}最新日志片段 (最后 10 行):${RESET}"
        if [ -f "$LOG_FILE" ]; then
            tail -n 10 "$LOG_FILE"
        fi
        [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"
        return 1
    fi
}

# 停止服务
stop_service() {
    echo -e "${CYAN}正在停止 Agent 服务...${RESET}"
    if ! is_running; then
        echo -e "${YELLOW}Agent 服务当前未在运行。${RESET}"
        [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo "向进程 (PID: ${pid}) 发送停止信号 (SIGTERM)..."
    kill "$pid" 2>/dev/null

    # 循环等待进程优雅退出，最长等待 5 秒
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 10 ]; do
        sleep 0.5
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}进程未在预期时间内响应，执行强制终止 (SIGKILL)...${RESET}"
        kill -9 "$pid" 2>/dev/null
        sleep 0.5
    fi

    [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Agent 服务已成功停止。${RESET}"
}

# 查看状态
status_service() {
    echo -e "${CYAN}=== Agent 服务状态 ===${RESET}"
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo -e "服务状态: ${GREEN}${BOLD}运行中 (Running)${RESET}"
        echo -e "进程 PID: ${GREEN}${pid}${RESET}"

        # 获取运行时长
        local uptime_info
        uptime_info=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        if [ -n "$uptime_info" ]; then
            echo -e "运行时长: ${uptime_info}"
        fi

        # 获取内存占用
        local mem_info
        mem_info=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
        if [ -n "$mem_info" ]; then
            local mem_mb=$((mem_info / 1024))
            echo -e "内存占用: ~${mem_mb} MB"
        fi

        echo -e "\n${CYAN}最新服务日志预览 (最后 8 行):${RESET}"
        if [ -f "$LOG_FILE" ]; then
            tail -n 8 "$LOG_FILE"
        else
            echo "  (暂无日志文件)"
        fi
    else
        echo -e "服务状态: ${RED}${BOLD}未运行 (Stopped)${RESET}"
        [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo -e "历史日志: ${LOG_FILE}"
        fi
    fi

    echo -e "\n${CYAN}=== 模型与下载状态 ===${RESET}"
    if is_download_running; then
        local dpid
        dpid=$(cat "$DOWNLOAD_PID_FILE")
        local d_model="未知"
        local d_pkg="未知"
        if [ -f "$DOWNLOAD_INFO_FILE" ]; then
            d_model=$(grep '^MODEL_ID=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
            d_pkg=$(grep '^PKG_DIR=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
        fi
        local d_uptime
        d_uptime=$(ps -p "$dpid" -o etime= 2>/dev/null | tr -d ' ')
        echo -e "下载任务: ${GREEN}${BOLD}正在后台下载中 (Downloading)${RESET}"
        echo -e "  下载模型: ${BOLD}${d_model}${RESET}"
        echo -e "  存储目录: models/${d_pkg}"
        echo -e "  进程 PID: ${dpid} (已运行 ${d_uptime:-'不久'})"
        echo -e "  (可在菜单选择【查看下载进度】查看实时日志流)"
    else
        echo -e "下载任务: ${YELLOW}当前无后台下载任务在运行${RESET}"
        [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"
    fi

    echo -e "\n本地已安装模型 (models/ 目录):"
    list_local_models
}

# 重启服务
restart_service() {
    stop_service
    sleep 1
    start_service
}

# 执行模型下载底层调度 (支持后台/前台)
do_download() {
    local model_id="$1"
    local pkg_dir="$2"
    local endpoint="${3:-https://huggingface.co}"
    local mode="${4:-bg}" # bg (后台) 或 fg (前台)

    if is_download_running; then
        local pid
        pid=$(cat "$DOWNLOAD_PID_FILE")
        echo -e "${YELLOW}当前已有下载任务在后台运行中 (PID: ${pid})，请勿重复启动。${RESET}"
        echo -e "您可选择【查看下载进度】或【停止下载任务】。"
        return 1
    fi

    [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"
    mkdir -p "$MODELS_DIR"

    # 记录元数据
    cat <<EOF > "$DOWNLOAD_INFO_FILE"
MODEL_ID=${model_id}
PKG_DIR=${pkg_dir}
ENDPOINT=${endpoint}
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
MODE=${mode}
EOF

    # 写头部日志
    {
        echo ""
        echo "========================================================"
        echo "下载任务启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "模型 ID: ${model_id}"
        echo "目标目录: models/${pkg_dir}"
        echo "下载源: ${endpoint}"
        echo "下载模式: ${mode}"
        echo "========================================================"
    } >> "$DOWNLOAD_LOG_FILE"

    if [ "$mode" = "bg" ]; then
        echo -e "${CYAN}正在启动后台下载任务...${RESET}"
        echo -e "  模型 ID : ${BOLD}${model_id}${RESET}"
        echo -e "  存储目录: ${BOLD}models/${pkg_dir}${RESET}"
        echo -e "  下载源  : ${endpoint}"

        # 启动后台任务 (设置 PYTHONUNBUFFERED=1 确保进度实时写入日志)
        nohup env HF_ENDPOINT="$endpoint" PYTHONUNBUFFERED=1 \
            "${SCRIPT_DIR}/scripts/download_model.sh" "$model_id" "$pkg_dir" >> "$DOWNLOAD_LOG_FILE" 2>&1 &
        local new_pid=$!
        echo "$new_pid" > "$DOWNLOAD_PID_FILE"

        sleep 1
        if kill -0 "$new_pid" 2>/dev/null; then
            echo -e "${GREEN}✓ 后台下载任务已成功启动！(PID: ${new_pid})${RESET}"
            echo -e "  日志文件: ${DOWNLOAD_LOG_FILE}"
            echo -e "  提示: 您可以在主菜单随时选择【查看下载进度】查看实时动态。\n"
        else
            echo -e "${RED}✗ 下载任务启动失败，请检查日志:${RESET}"
            tail -n 12 "$DOWNLOAD_LOG_FILE"
            [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"
            return 1
        fi
    else
        echo -e "${CYAN}正在启动前台下载任务 (按 Ctrl+C 可随时中断，下次自动断点续传)...${RESET}"
        echo -e "  模型 ID : ${BOLD}${model_id}${RESET}"
        echo -e "  存储目录: ${BOLD}models/${pkg_dir}${RESET}"
        echo -e "  下载源  : ${endpoint}\n"

        env HF_ENDPOINT="$endpoint" \
            "${SCRIPT_DIR}/scripts/download_model.sh" "$model_id" "$pkg_dir" 2>&1 | tee -a "$DOWNLOAD_LOG_FILE"
    fi
}

# 下载模型交互菜单
download_model_menu() {
    echo -e "\n${BOLD}${CYAN}=== 下载 ASR 模型 ===${RESET}"
    echo -e "当前本地已安装模型:"
    list_local_models
    echo ""

    if is_download_running; then
        local pid
        pid=$(cat "$DOWNLOAD_PID_FILE")
        echo -e "${YELLOW}警告: 当前已有模型正在后台下载中 (PID: ${pid})！${RESET}"
        echo -e "请先在主菜单选择【查看下载进度】或【停止下载任务】后再下载新模型。\n"
        read -r -p "按回车键返回主菜单..." _
        return 0
    fi

    echo "请选择要下载的模型:"
    echo -e "  ${BOLD}1)${RESET} Qwen3-ASR-0.6B    ${GREEN}[推荐/平台默认]${RESET} (Qwen/Qwen3-ASR-0.6B -> qwen3-asr-0.6b, ~1.8GB)"
    echo -e "  ${BOLD}2)${RESET} Qwen3-ASR-1.7B    ${CYAN}[高精度]${RESET}       (Qwen/Qwen3-ASR-1.7B -> qwen3-asr-1.7b, ~4.5GB)"
    echo -e "  ${BOLD}3)${RESET} Whisper-base      ${YELLOW}[轻量英文]${RESET}     (openai/whisper-base -> whisper-base, ~145MB)"
    echo -e "  ${BOLD}4)${RESET} Whisper-small     ${YELLOW}[中等模型]${RESET}     (openai/whisper-small -> whisper-small, ~480MB)"
    echo -e "  ${BOLD}5)${RESET} Whisper-large-v3  ${CYAN}[高精度多语]${RESET}   (openai/whisper-large-v3 -> whisper-large-v3, ~3.1GB)"
    echo -e "  ${BOLD}6)${RESET} 自定义 Hugging Face 模型"
    echo -e "  ${BOLD}0)${RESET} 返回上级菜单"
    echo ""

    read -r -p "请输入模型编号 (0-6): " choice
    local model_id=""
    local pkg_dir=""

    case "$choice" in
        1)
            model_id="Qwen/Qwen3-ASR-0.6B"
            pkg_dir="qwen3-asr-0.6b"
            ;;
        2)
            model_id="Qwen/Qwen3-ASR-1.7B"
            pkg_dir="qwen3-asr-1.7b"
            ;;
        3)
            model_id="openai/whisper-base"
            pkg_dir="whisper-base"
            ;;
        4)
            model_id="openai/whisper-small"
            pkg_dir="whisper-small"
            ;;
        5)
            model_id="openai/whisper-large-v3"
            pkg_dir="whisper-large-v3"
            ;;
        6)
            read -r -p "请输入 Hugging Face Model ID (例如 Qwen/Qwen3-ASR-0.6B): " custom_id
            if [ -z "$custom_id" ]; then
                echo -e "${RED}模型 ID 不能为空。${RESET}"
                return 1
            fi
            local default_pkg
            default_pkg=$(echo "${custom_id##*/}" | tr '[:upper:]' '[:lower:]')
            read -r -p "请输入本地存储目录名 (默认: ${default_pkg}): " custom_pkg
            pkg_dir="${custom_pkg:-$default_pkg}"
            model_id="$custom_id"
            ;;
        0|"")
            return 0
            ;;
        *)
            echo -e "${RED}无效的选择。${RESET}"
            return 1
            ;;
    esac

    # 检查是否使用镜像加速
    local endpoint="https://hf-mirror.com"
    if [ -n "${HF_ENDPOINT:-}" ]; then
        endpoint="$HF_ENDPOINT"
    fi

    echo ""
    read -r -p "是否使用国内镜像加速 (https://hf-mirror.com)? [Y/n] (默认: Y): " mirror_choice
    case "$mirror_choice" in
        [nN][oO]|[nN])
            endpoint="https://huggingface.co"
            ;;
        *)
            endpoint="https://hf-mirror.com"
            ;;
    esac

    # 检查下载方式 (前台/后台)
    echo ""
    echo "请选择下载运行方式:"
    echo -e "  ${BOLD}1)${RESET} 后台下载 ${GREEN}[推荐]${RESET} (后台静默下载，不阻塞终端，可随时查看进度)"
    echo -e "  ${BOLD}2)${RESET} 前台下载 (在当前窗口实时显示下载进度条)"
    echo -e "  ${BOLD}0)${RESET} 取消"
    read -r -p "请输入运行方式编号 (0-2, 默认: 1): " mode_choice

    local mode="bg"
    case "$mode_choice" in
        2)
            mode="fg"
            ;;
        0)
            echo -e "${YELLOW}已取消下载。${RESET}"
            return 0
            ;;
        *)
            mode="bg"
            ;;
    esac

    do_download "$model_id" "$pkg_dir" "$endpoint" "$mode"
}

# 查看下载进度与日志追踪
view_download_progress() {
    echo -e "\n${BOLD}${CYAN}=== 模型下载状态与进度 ===${RESET}"

    local model_id="未知"
    local pkg_dir="未知"
    local start_time="未知"
    local endpoint="未知"
    if [ -f "$DOWNLOAD_INFO_FILE" ]; then
        model_id=$(grep '^MODEL_ID=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
        pkg_dir=$(grep '^PKG_DIR=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
        start_time=$(grep '^START_TIME=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
        endpoint=$(grep '^ENDPOINT=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
    fi

    if is_download_running; then
        local pid
        pid=$(cat "$DOWNLOAD_PID_FILE")
        local uptime_info
        uptime_info=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')

        echo -e "运行状态: ${GREEN}${BOLD}正在后台下载中 (Downloading)${RESET}"
        echo -e "下载进程: PID ${GREEN}${pid}${RESET} (已运行 ${uptime_info:-'未知'})"
        echo -e "当前模型: ${BOLD}${model_id}${RESET}"
        echo -e "存储路径: ${BOLD}models/${pkg_dir}${RESET}"
        echo -e "下载来源: ${endpoint}"
        echo -e "启动时间: ${start_time}"

        local target_dir="${MODELS_DIR}/${pkg_dir}"
        if [ -d "$target_dir" ]; then
            local disk_usage
            disk_usage=$(du -sh "$target_dir" 2>/dev/null | awk '{print $1}')
            echo -e "当前磁盘写入: ${disk_usage}"
        fi

        echo -e "\n${CYAN}最新下载日志预览 (最后 12 行):${RESET}"
        if [ -f "$DOWNLOAD_LOG_FILE" ]; then
            tail -n 12 "$DOWNLOAD_LOG_FILE"
        else
            echo "  (暂无日志文件)"
        fi

        echo ""
        read -r -p "是否进入实时日志流追踪? (按 Ctrl+C 可随时退出追踪返回菜单) [Y/n]: " follow_opt
        case "$follow_opt" in
            [nN][oO]|[nN])
                ;;
            *)
                echo -e "\n${CYAN}--- 进入下载日志追踪 (按 Ctrl+C 退出追踪，下载仍在后台继续) ---${RESET}"
                trap 'echo -e "\n${GREEN}已退出日志追踪。${RESET}"' INT
                tail -n 25 -f "$DOWNLOAD_LOG_FILE" 2>/dev/null || true
                trap - INT
                ;;
        esac
    else
        echo -e "运行状态: ${YELLOW}当前无后台下载任务在运行${RESET}"
        [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"

        if [ -f "$DOWNLOAD_LOG_FILE" ]; then
            echo -e "上次任务: ${model_id} -> models/${pkg_dir} (启动于 ${start_time})"
            echo -e "\n${CYAN}最近下载日志末尾 (最后 12 行):${RESET}"
            tail -n 12 "$DOWNLOAD_LOG_FILE"
        fi

        echo -e "\n${CYAN}已下载的模型列表 (models/ 目录):${RESET}"
        list_local_models
        echo ""
        read -r -p "按回车键返回主菜单..." _
    fi
}

# 停止下载任务
stop_download() {
    echo -e "\n${BOLD}${CYAN}=== 停止下载任务 ===${RESET}"
    if ! is_download_running; then
        echo -e "${YELLOW}当前没有正在运行的后台下载任务。${RESET}"
        [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$DOWNLOAD_PID_FILE")
    local model_id="未知"
    if [ -f "$DOWNLOAD_INFO_FILE" ]; then
        model_id=$(grep '^MODEL_ID=' "$DOWNLOAD_INFO_FILE" | cut -d'=' -f2-)
    fi

    read -r -p "确认停止模型 [${model_id}] 的后台下载 (PID: ${pid})? [y/N]: " confirm
    case "$confirm" in
        [yY][eE][sS]|[yY])
            ;;
        *)
            echo -e "${YELLOW}已取消操作。${RESET}"
            return 0
            ;;
    esac

    echo "正在向下载进程 (PID: ${pid}) 发送优雅终止信号 (SIGINT)..."
    kill -INT "$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    pkill -INT -P "$pid" 2>/dev/null || true

    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 10 ]; do
        sleep 0.5
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}下载进程仍在运行，发送强制终止信号 (SIGKILL)...${RESET}"
        kill -9 "$pid" 2>/dev/null
        pkill -9 -P "$pid" 2>/dev/null || true
        sleep 0.5
    fi

    [ -f "$DOWNLOAD_PID_FILE" ] && /bin/rm -f "$DOWNLOAD_PID_FILE"
    echo -e "${GREEN}✓ 下载任务已停止。${RESET}"
    echo -e "  提示: 底层支持断点续传，已下载的部分数据已保留，下次重新下载将直接从断点继续。"
}

# 交互式菜单逻辑
interactive_menu() {
    PS3="请选择操作编号: "
    options=(
        "启动服务"
        "停止服务"
        "重启服务"
        "查看服务状态"
        "下载模型"
        "查看下载进度"
        "停止下载任务"
        "退出"
    )

    while true; do
        echo -e "\n${BOLD}=== Cyphr Agent 管理面板 ===${RESET}"
        select opt in "${options[@]}"; do
            case $REPLY in
                1)
                    start_service
                    break
                    ;;
                2)
                    stop_service
                    break
                    ;;
                3)
                    restart_service
                    break
                    ;;
                4)
                    status_service
                    break
                    ;;
                5)
                    download_model_menu
                    break
                    ;;
                6)
                    view_download_progress
                    break
                    ;;
                7)
                    stop_download
                    break
                    ;;
                8)
                    echo -e "${GREEN}再见！${RESET}"
                    exit 0
                    ;;
                *)
                    echo -e "${RED}无效的选择，请重新输入（1-${#options[@]}）。${RESET}"
                    ;;
            esac
        done
    done
}

# 支持命令行参数直接调用，无参数时启动交互式菜单
case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    download)
        if [ -n "${2:-}" ]; then
            local_model="$2"
            local_pkg="${3:-${local_model##*/}}"
            local_pkg=$(echo "$local_pkg" | tr '[:upper:]' '[:lower:]')
            local_endpoint="${4:-https://hf-mirror.com}"
            local_mode="${5:-bg}"
            do_download "$local_model" "$local_pkg" "$local_endpoint" "$local_mode"
        else
            download_model_menu
        fi
        ;;
    progress)
        view_download_progress
        ;;
    stop-download)
        stop_download
        ;;
    models)
        echo -e "${CYAN}=== 本地已安装模型列表 ===${RESET}"
        list_local_models
        ;;
    -h|--help|help)
        echo "Cyphr Agent 管理脚本使用帮助"
        echo ""
        echo "用法: ./agent.sh [命令]"
        echo ""
        echo "可用命令:"
        echo "  (无参数)       进入交互式管理面板菜单"
        echo "  start          后台启动 Agent 服务"
        echo "  stop           停止 Agent 服务"
        echo "  restart        重启 Agent 服务"
        echo "  status         查看 Agent 服务状态、下载任务及模型列表"
        echo "  download       进入模型下载菜单，或指定参数: download <MODEL_ID> [PKG_DIR] [ENDPOINT] [MODE]"
        echo "  progress       查看当前模型下载进度及实时日志"
        echo "  stop-download  停止当前正在运行的后台模型下载任务"
        echo "  models         列出本地已下载的模型包"
        echo "  help           显示此帮助信息"
        ;;
    *)
        interactive_menu
        ;;
esac
