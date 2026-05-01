#!/bin/bash
# Fish Speech 多实例启动脚本

set -e

FISH_SPEECH_DIR="/disk0/repo/manju/fish-speech"
MANAGER_SCRIPT="$FISH_SPEECH_DIR/fish_speech_manager.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助
show_help() {
    cat << EOF
Fish Speech 多实例管理工具

用法: $0 [命令] [选项]

命令:
    start              启动 Fish Speech 实例
    stop               停止 Fish Speech 实例
    status             显示实例状态
    test               测试语音生成
    restart            重启实例

启动选项:
    --single GPU_ID    启动单GPU实例（例如: --single 0）
    --multi [NUM]      启动多GPU实例（可选指定数量，例如: --multi 2）
                      不指定数量则自动使用所有可用GPU

停止选项:
    --instance ID      停止指定实例ID
    --all              停止所有实例

测试选项:
    -t TEXT            要生成的文本
    -r REFERENCE_ID    音色ID
    -o OUTPUT          输出路径

示例:
    # 启动单GPU实例（使用GPU 0）
    $0 start --single 0

    # 启动多GPU实例（使用所有可用GPU）
    $0 start --multi

    # 启动多GPU实例（只使用2张卡）
    $0 start --multi 2

    # 查看实例状态
    $0 status

    # 停止所有实例
    $0 stop --all

    # 停止指定实例
    $0 stop --instance 1

    # 测试语音生成
    $0 test -t "你好" -r young_girl_1 -o /tmp/test.wav

    # 重启所有实例
    $0 restart --multi 2

EOF
}

# 检查依赖
check_dependencies() {
    if [ ! -d "$FISH_SPEECH_DIR" ]; then
        print_error "Fish Speech 目录不存在: $FISH_SPEECH_DIR"
        exit 1
    fi

    if [ ! -f "$MANAGER_SCRIPT" ]; then
        print_error "管理脚本不存在: $MANAGER_SCRIPT"
        exit 1
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未检测到 NVIDIA 驱动"
        exit 1
    fi
}

# 主逻辑
main() {
    check_dependencies

    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        start)
            shift
            cd "$FISH_SPEECH_DIR"
            python fish_speech_manager.py start "$@"
            ;;

        stop)
            shift
            cd "$FISH_SPEECH_DIR"
            python fish_speech_manager.py stop "$@"
            ;;

        status)
            cd "$FISH_SPEECH_DIR"
            python fish_speech_manager.py status
            ;;

        test)
            shift
            cd "$FISH_SPEECH_DIR"
            python fish_speech_manager.py test "$@"
            ;;

        restart)
            shift
            print_info "停止所有实例..."
            cd "$FISH_SPEECH_DIR"
            python fish_speech_manager.py stop --all
            sleep 2
            print_info "启动实例..."
            python fish_speech_manager.py start "$@"
            ;;

        -h|--help)
            show_help
            ;;

        *)
            print_error "未知命令: $1"
            echo
            show_help
            exit 1
            ;;
    esac
}

main "$@"
