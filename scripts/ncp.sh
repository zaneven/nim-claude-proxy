#!/usr/bin/env bash
# nim-claude-proxy (ncp) daemon management & CLI tool

set -e

# Resolve actual project root directory even when executed via symlink
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." >/dev/null 2>&1 && pwd )"

LOGS_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOGS_DIR/ncp.pid"
LOG_FILE="$LOGS_DIR/server.log"
ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"

# Helper to read .env variable
get_env_val() {
  local key="$1"
  local default_val="$2"
  if [ -f "$ENV_FILE" ]; then
    local val
    val=$(grep -E "^${key}=" "$ENV_FILE" | cut -d '=' -f2- | tr -d '\r' | tr -d '"' | tr -d "'")
    if [ -n "$val" ]; then
      echo "$val"
      return
    fi
  fi
  echo "$default_val"
}

HOST=$(get_env_val "HOST" "0.0.0.0")
PORT=$(get_env_val "PORT" "8000")

mkdir -p "$LOGS_DIR"

show_help() {
  echo "================================================================="
  echo "        NIM Claude Proxy (ncp) 服务管理与 CLI 工具"
  echo "================================================================="
  echo "用法:"
  echo "  ncp <command>"
  echo ""
  echo "可用命令:"
  echo "  install    初始化项目依赖，配置 .env，并注册 ncp 命令到全局 PATH"
  echo "  start      在后台启动 NIM Proxy 服务"
  echo "  stop       停止正在运行的 NIM Proxy 服务"
  echo "  restart    重启 NIM Proxy 服务"
  echo "  status     查看 NIM Proxy 服务运行状态及健康检测"
  echo "  help       显示当前帮助页面"
  echo "================================================================="
}

is_running() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  # Secondary check: check if process matching nim_proxy.main or nim-proxy is running
  local ps_pid
  ps_pid=$(pgrep -f "nim_proxy.main|nim-proxy" | head -n 1 || true)
  if [ -n "$ps_pid" ]; then
    echo "$ps_pid" > "$PID_FILE"
    echo "$ps_pid"
    return 0
  fi
  return 1
}

cmd_install() {
  echo "[INFO] 开始安装并配置 NIM Proxy (ncp)..."
  
  # 1. 确保 .env 文件存在
  if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
      cp "$ENV_EXAMPLE" "$ENV_FILE"
      echo "[INFO] 已根据 .env.example 创建默认 .env 文件."
    else
      touch "$ENV_FILE"
      echo "[INFO] 已创建空的 .env 文件."
    fi
  else
    echo "[INFO] 检测到已有 .env 配置文件."
  fi

  # 2. 确保 uv / 依赖就绪
  if command -v uv >/dev/null 2>&1; then
    echo "[INFO] 正在初始化 Python 依赖环境 (uv sync)..."
    (cd "$PROJECT_DIR" && uv sync)
  else
    echo "[WARN] 未检测到 'uv' 命令，请确保 Python 虚拟环境与依赖已正确手动安装。"
  fi

  # 3. 设置脚本执行权限
  chmod +x "$PROJECT_DIR/scripts/ncp.sh" 2>/dev/null || true
  if [ -f "$PROJECT_DIR/scripts/install.sh" ]; then
    chmod +x "$PROJECT_DIR/scripts/install.sh" 2>/dev/null || true
  fi

  # 4. 注册全局命令 ncp
  local script_path="$PROJECT_DIR/scripts/ncp.sh"
  local installed=false
  local target_dirs=("$HOME/.local/bin" "/usr/local/bin" "/opt/homebrew/bin")

  for dir in "${target_dirs[@]}"; do
    if [ -d "$dir" ] && [ -w "$dir" ]; then
      ln -sf "$script_path" "$dir/ncp"
      echo "[SUCCESS] 已成功在 $dir/ncp 创建全局软链接."
      installed=true
      break
    fi
  done

  if [ "$installed" = false ]; then
    mkdir -p "$HOME/.local/bin" 2>/dev/null || true
    if [ -d "$HOME/.local/bin" ] && [ -w "$HOME/.local/bin" ]; then
      ln -sf "$script_path" "$HOME/.local/bin/ncp"
      echo "[SUCCESS] 已成功在 $HOME/.local/bin/ncp 创建全局软链接."
      installed=true
    else
      echo "[WARN] 无法直接写入全局目录。请使用 sudo 或手动运行:"
      echo "       sudo ln -sf \"$script_path\" /usr/local/bin/ncp"
    fi
  fi

  echo "-----------------------------------------------------------------"
  echo "[SUCCESS] 安装与全局命令注册完成！"
  echo "现在可以在任意终端窗口直接使用以下命令："
  echo "  - ncp start    : 启动服务"
  echo "  - ncp status   : 查看状态"
  echo "  - ncp stop     : 停止服务"
  echo "-----------------------------------------------------------------"
}

cmd_start() {
  local pid
  if pid=$(is_running); then
    echo "[WARN] NIM Proxy 已经在运行中 (PID: $pid)."
    echo "[INFO] 管理后台: http://localhost:$PORT/admin"
    return 0
  fi

  echo "[INFO] 正在启动 NIM Proxy 服务..."
  
  cd "$PROJECT_DIR"
  if command -v uv >/dev/null 2>&1; then
    nohup uv run nim-proxy >> "$LOG_FILE" 2>&1 &
  else
    nohup python -m nim_proxy.main >> "$LOG_FILE" 2>&1 &
  fi

  pid=$!
  echo "$pid" > "$PID_FILE"

  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "[SUCCESS] NIM Proxy 服务已成功启动!"
    echo "  - 进程 PID : $pid"
    echo "  - 端口监听 : $PORT"
    echo "  - 管理后台 : http://localhost:$PORT/admin"
    echo "  - 代理 API  : http://localhost:$PORT/v1/messages"
    echo "  - 日志路径 : $LOG_FILE"
  else
    echo "[ERROR] 服务启动失败，请检查日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
  fi
}

cmd_stop() {
  local pid
  if pid=$(is_running); then
    echo "[INFO] 正在停止 NIM Proxy 服务 (PID: $pid)..."
    kill "$pid" 2>/dev/null || true

    local count=0
    while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 10 ]; do
      sleep 1
      count=$((count + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
      echo "[WARN] 优雅停止超时，强制终止进程 (SIGKILL)..."
      kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo "[SUCCESS] NIM Proxy 服务已成功停止."
  else
    echo "[INFO] NIM Proxy 服务未在运行."
    rm -f "$PID_FILE" 2>/dev/null || true
  fi
}

cmd_restart() {
  echo "[INFO] 正在重启 NIM Proxy 服务..."
  cmd_stop
  sleep 1
  cmd_start
}

cmd_status() {
  echo "================================================================="
  echo "                 NIM Proxy 服务运行状态"
  echo "================================================================="
  local pid
  if pid=$(is_running); then
    echo "  运行状态 : [ RUNNING ]"
    echo "  进程 PID : $pid"
    echo "  服务端口 : $PORT"
    echo "  管理后台 : http://localhost:$PORT/admin"
    echo "  代理 API  : http://localhost:$PORT/v1/messages"
    echo "  日志路径 : $LOG_FILE"
    echo "-----------------------------------------------------------------"
    
    if command -v curl >/dev/null 2>&1; then
      local http_code
      http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")
      if [ "$http_code" = "200" ]; then
        echo "  健康检测 : [ OK ] (HTTP 200)"
      else
        echo "  健康检测 : [ WARN ] (HTTP 返回码: $http_code)"
      fi
    fi
  else
    echo "  运行状态 : [ STOPPED ]"
    echo "  服务端口 : $PORT"
    echo "  日志路径 : $LOG_FILE"
  fi
  echo "================================================================="
}

case "${1:-}" in
  install)
    cmd_install
    ;;
  start)
    cmd_start
    ;;
  stop)
    cmd_stop
    ;;
  restart)
    cmd_restart
    ;;
  status)
    cmd_status
    ;;
  help|-h|--help)
    show_help
    ;;
  "")
    show_help
    ;;
  *)
    echo "[ERROR] 未知命令: '$1'"
    echo ""
    show_help
    exit 1
    ;;
esac
