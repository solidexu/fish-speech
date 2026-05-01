#!/usr/bin/env python3
"""
Fish Speech 同步多实例启动器
确保多个实例能够同时启动并正确分配资源
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

FISH_SPEECH_DIR = "/disk0/repo/manju/third_party/fish-speech"
CONFIG_FILE = os.path.join(FISH_SPEECH_DIR, "multi_instance_config.json")


def start_instances(gpus):
    """同时启动多个实例"""
    processes = []

    for i, gpu_id in enumerate(gpus):
        instance_id = i + 1
        port = 9997 + i
        log_file = f"/tmp/fish_speech_instances/instance_{instance_id}_gpu{gpu_id}.log"

        cmd = [
            "python3",  # 使用系统 Python3
            f"{FISH_SPEECH_DIR}/tools/api_server.py",
            "--llama-checkpoint-path", "checkpoints/s2-pro",
            "--decoder-checkpoint-path", "checkpoints/s2-pro/codec.pth",
            "--listen", f"0.0.0.0:{port}",
            "--half"
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # 确保日志目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            print(f"启动实例 {instance_id} 在 GPU {gpu_id}, 端口 {port}")
            process = subprocess.Popen(
                cmd,
                cwd=FISH_SPEECH_DIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            processes.append({
                'instance_id': instance_id,
                'gpu_id': gpu_id,
                'port': port,
                'pid': process.pid,
                'process': process
            })

    return processes


def wait_for_ready(processes, timeout=90):
    """等待所有实例就绪"""
    print(f"\n等待 {len(processes)} 个实例启动...")

    import requests

    start_time = time.time()
    ready_count = 0

    while time.time() - start_time < timeout:
        ready_count = 0
        for proc in processes:
            port = proc['port']
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=1)
                if response.status_code == 200:
                    ready_count += 1
            except:
                pass

        print(f"\r就绪: {ready_count}/{len(processes)} 个实例", end='', flush=True)
        time.sleep(2)

    print()
    return ready_count == len(processes)


def save_config(processes):
    """保存配置"""
    config = {
        'instances': [
            {
                'instance_id': p['instance_id'],
                'gpu_id': p['gpu_id'],
                'port': p['port'],
                'pid': p['pid'],
                'status': 'running'
            }
            for p in processes
        ]
    }

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def stop_all():
    """停止所有实例"""
    if not os.path.exists(CONFIG_FILE):
        return

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    for inst in config.get('instances', []):
        pid = inst.get('pid')
        if pid:
            try:
                os.killpg(os.getpgid(pid), 9)  # SIGKILL
                print(f"已停止实例 {inst['instance_id']} (PID: {pid})")
            except:
                pass

    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fish Speech 同步启动器')
    parser.add_argument('action', choices=['start', 'stop', 'status'], help='动作')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU列表，逗号分隔')

    args = parser.parse_args()

    if args.action == 'stop':
        stop_all()
        return

    if args.action == 'status':
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                print("\nFish Speech 实例状态:")
                print("-" * 70)
                print(f"{'ID':<4} {'GPU':<4} {'端口':<6} {'PID':<8} {'状态':<10}")
                print("-" * 70)
                for inst in config.get('instances', []):
                    print(f"{inst['instance_id']:<4} {inst['gpu_id']:<4} {inst['port']:<6} {inst['pid']:<8} {inst['status']:<10}")
                print("-" * 70)
        else:
            print("当前没有运行的实例")
        return

    if args.action == 'start':
        gpus = [int(x.strip()) for x in args.gpus.split(',')]

        print(f"=== 同时启动 {len(gpus)} 个实例 ===")
        print(f"GPU: {gpus}")

        processes = start_instances(gpus)

        # 等待就绪
        if wait_for_ready(processes):
            print("\n✓ 所有实例启动成功")
            save_config(processes)
            print("\n实例信息:")
            for p in processes:
                print(f"  实例 {p['instance_id']}: GPU {p['gpu_id']}, 端口 {p['port']}, PID {p['pid']}")
        else:
            print("\n✗ 部分实例启动超时")
            # 显示哪个实例没启动成功
            for p in processes:
                try:
                    import requests
                    response = requests.get(f"http://localhost:{p['port']}/", timeout=1)
                    status = "✓ 运行中"
                except:
                    status = "✗ 未就绪"
                print(f"  实例 {p['instance_id']}: {status}")


if __name__ == '__main__':
    main()
