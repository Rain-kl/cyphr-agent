#!/bin/bash

# Copyright 2026 Arctel.net
# SPDX-License-Identifier: Apache-2.0

# 确保以脚本所在目录为基准路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PID_FILE="${SCRIPT_DIR}/agent.pid"
LOG_FILE="${SCRIPT_DIR}/agent.log"
ENV_FILE="${SCRIPT_DIR}/.env"

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

# 检查进程是否正在运行
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
        echo -e "当前状态: ${GREEN}${BOLD}运行中 (Running)${RESET}"
        echo -e "进程 PID: ${GREEN}${pid}${RESET}"

        # 尝试获取进程启动时间与运行时间
        local uptime_info
        uptime_info=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        if [ -n "$uptime_info" ]; then
            echo -e "运行时长: ${uptime_info}"
        fi

        # 尝试获取内存占用
        local mem_info
        mem_info=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ')
        if [ -n "$mem_info" ]; then
            local mem_mb=$((mem_info / 1024))
            echo -e "内存占用: ~${mem_mb} MB"
        fi

        echo -e "\n${CYAN}最新日志预览 (最后 8 行):${RESET}"
        if [ -f "$LOG_FILE" ]; then
            tail -n 8 "$LOG_FILE"
        else
            echo "  (暂无日志文件)"
        fi
    else
        echo -e "当前状态: ${RED}${BOLD}未运行 (Stopped)${RESET}"
        [ -f "$PID_FILE" ] && /bin/rm -f "$PID_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo -e "历史日志位置: ${LOG_FILE}"
        fi
    fi
}

# 重启服务
restart_service() {
    stop_service
    sleep 1
    start_service
}

# 交互式菜单逻辑
interactive_menu() {
    # 设置提示符
    PS3="请选择操作编号: "

    # 定义菜单选项
    options=("启动服务" "停止服务" "查看状态" "退出")

    echo -e "\n${BOLD}=== Cyphr Agent 管理面板 ===${RESET}"

    select opt in "${options[@]}"; do
        case $REPLY in
            1)
                start_service
                ;;
            2)
                stop_service
                ;;
            3)
                status_service
                ;;
            4)
                echo -e "${GREEN}再见！${RESET}"
                break
                ;;
            *)
                echo -e "${RED}无效的选择，请重新输入（1-${#options[@]}）。${RESET}"
                ;;
        esac
        # 清空 REPLY 以便下一次循环重新提示
        REPLY=
    done
}

# 支持命令行参数直接调用，无参数时启动交互式菜单
case "$1" in
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
    *)
        interactive_menu
        ;;
esac
