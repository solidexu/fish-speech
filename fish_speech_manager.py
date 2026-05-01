#!/usr/bin/env python3
"""
Fish Speech 多实例管理器
支持单卡和多卡并发模式，提高语音生成吞吐量
"""

import os
import subprocess
import signal
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import aiohttp
import requests


@dataclass
class FishSpeechInstance:
    """Fish Speech 实例配置"""
    instance_id: int
    gpu_id: int
    port: int
    pid: Optional[int] = None
    status: str = "stopped"  # stopped, starting, running, error

    def to_dict(self) -> dict:
        return asdict(self)


class FishSpeechManager:
    """Fish Speech 多实例管理器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/disk0/repo/manju/fish-speech/multi_instance_config.json"
        self.fish_speech_dir = "/disk0/repo/manju/fish-speech"
        self.log_dir = Path("/tmp/fish_speech_instances")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.instances: Dict[int, FishSpeechInstance] = {}
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for inst_data in config.get('instances', []):
                    inst = FishSpeechInstance(**inst_data)
                    self.instances[inst.instance_id] = inst

    def save_config(self):
        """保存配置文件"""
        config = {
            'instances': [inst.to_dict() for inst in self.instances.values()]
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # 强制写入磁盘

    def get_available_gpus(self) -> List[int]:
        """获取可用GPU列表"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_id = int(parts[0])
                    mem_free = int(parts[1])
                    mem_total = int(parts[2])
                    # 至少需要 23GB 可用显存（模型占用约22GB，需要一些buffer）
                    if mem_free >= 23000:
                        gpus.append(gpu_id)
            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"获取GPU信息失败: {e}")
            return []

    def start_instance(self, instance_id: int, gpu_id: int, port: int) -> bool:
        """启动单个 Fish Speech 实例"""
        if instance_id in self.instances and self.instances[instance_id].status == "running":
            print(f"实例 {instance_id} 已在运行")
            return True

        log_file = self.log_dir / f"instance_{instance_id}_gpu{gpu_id}.log"

        # 启动命令
        cmd = [
            os.path.join(self.fish_speech_dir, '.venv/bin/python'),
            'tools/api_server.py',
            '--llama-checkpoint-path', 'checkpoints/s2-pro',
            '--decoder-checkpoint-path', 'checkpoints/s2-pro/codec.pth',
            '--listen', f'0.0.0.0:{port}',
            '--half'
        ]

        # 设置GPU环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            # 启动进程
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=self.fish_speech_dir,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )

            # 保存实例信息
            instance = FishSpeechInstance(
                instance_id=instance_id,
                gpu_id=gpu_id,
                port=port,
                pid=process.pid,
                status="starting"
            )
            self.instances[instance_id] = instance
            self.save_config()

            print(f"✓ 启动实例 {instance_id}: GPU {gpu_id}, 端口 {port}, PID {process.pid}")
            print(f"  日志: {log_file}")

            # 等待服务启动
            return self._wait_for_instance_ready(instance, timeout=60)
        except Exception as e:
            print(f"✗ 启动实例 {instance_id} 失败: {e}")
            if instance_id in self.instances:
                self.instances[instance_id].status = "error"
                self.save_config()
            return False

    def _wait_for_instance_ready(self, instance: FishSpeechInstance, timeout: int = 60) -> bool:
        """等待实例就绪"""
        # 尝试访问根路径或swagger页面来判断服务是否启动
        url = f"http://localhost:{instance.port}/"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    instance.status = "running"
                    self.save_config()
                    print(f"✓ 实例 {instance.instance_id} 启动成功")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)
            # 检查进程是否存活
            if not self._is_process_alive(instance.pid):
                instance.status = "error"
                self.save_config()
                print(f"✗ 实例 {instance.instance_id} 进程已退出")
                return False

        instance.status = "error"
        self.save_config()
        print(f"✗ 实例 {instance.instance_id} 启动超时")
        return False

    def _is_process_alive(self, pid: int) -> bool:
        """检查进程是否存活"""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def stop_instance(self, instance_id: int) -> bool:
        """停止单个实例"""
        if instance_id not in self.instances:
            print(f"实例 {instance_id} 不存在")
            return False

        instance = self.instances[instance_id]

        if instance.status != "running" and instance.status != "starting":
            print(f"实例 {instance_id} 状态为 {instance.status}，无需停止")
            return True

        try:
            # 发送 SIGTERM 给进程组
            if instance.pid:
                os.killpg(os.getpgid(instance.pid), signal.SIGTERM)

            # 等待进程结束
            for _ in range(10):
                if not self._is_process_alive(instance.pid):
                    break
                time.sleep(1)

            # 如果进程还活着，发送 SIGKILL
            if self._is_process_alive(instance.pid):
                os.killpg(os.getpgid(instance.pid), signal.SIGKILL)
                time.sleep(1)

            instance.status = "stopped"
            instance.pid = None
            self.save_config()
            print(f"✓ 已停止实例 {instance_id}")
            return True
        except Exception as e:
            print(f"✗ 停止实例 {instance_id} 失败: {e}")
            return False

    def stop_all(self) -> bool:
        """停止所有实例"""
        success = True
        for instance_id in list(self.instances.keys()):
            if not self.stop_instance(instance_id):
                success = False
        return success

    def status(self):
        """显示所有实例状态"""
        if not self.instances:
            print("当前没有配置的实例")
            return

        print("\nFish Speech 实例状态:")
        print("-" * 70)
        print(f"{'ID':<4} {'GPU':<4} {'端口':<6} {'PID':<8} {'状态':<10}")
        print("-" * 70)

        for inst in sorted(self.instances.values(), key=lambda x: x.instance_id):
            pid_str = str(inst.pid) if inst.pid else "-"
            print(f"{inst.instance_id:<4} {inst.gpu_id:<4} {inst.port:<6} {pid_str:<8} {inst.status:<10}")

        print("-" * 70)

        # 显示可用GPU
        available_gpus = self.get_available_gpus()
        if available_gpus:
            print(f"\n可用GPU: {available_gpus}")
        else:
            print("\n没有足够的GPU资源")

    def start_multi_gpu(self, num_gpus: Optional[int] = None):
        """启动多GPU实例"""
        available_gpus = self.get_available_gpus()

        if not available_gpus:
            print("没有可用的GPU（需要至少23GB可用显存）")
            return False

        if num_gpus:
            selected_gpus = available_gpus[:num_gpus]
        else:
            selected_gpus = available_gpus

        print(f"使用GPU: {selected_gpus}")

        success = True
        base_port = 9997

        # 收集正在运行的GPU
        running_gpus = set()
        for inst in self.instances.values():
            if inst.status == "running":
                running_gpus.add(inst.gpu_id)

        # 只为未运行的GPU启动新实例
        gpus_to_start = [gpu for gpu in selected_gpus if gpu not in running_gpus]

        if not gpus_to_start:
            print("所有GPU的实例已经在运行中")
            return True

        for gpu_id in gpus_to_start:
            # 找到当前可用的实例ID（跳过已使用的）
            instance_id = 1
            while instance_id in self.instances:
                instance_id += 1

            port = base_port + (instance_id - 1)

            print(f"\n启动实例 {instance_id} 在 GPU {gpu_id}...")
            if not self.start_instance(instance_id, gpu_id, port):
                success = False
                break

        return success

    def start_single_gpu(self, gpu_id: int = 0):
        """启动单GPU实例"""
        print(f"启动单GPU实例在 GPU {gpu_id}...")
        return self.start_instance(1, gpu_id, 9997)


class FishSpeechClient:
    """Fish Speech 客户端，支持负载均衡"""

    def __init__(self, instances: List[FishSpeechInstance]):
        self.instances = instances
        self.current_idx = 0

    def get_next_instance(self) -> Optional[FishSpeechInstance]:
        """获取下一个可用实例（轮询）"""
        running_instances = [inst for inst in self.instances if inst.status == "running"]

        if not running_instances:
            return None

        instance = running_instances[self.current_idx % len(running_instances)]
        self.current_idx += 1
        return instance

    async def generate_speech_async(
        self,
        text: str,
        reference_id: str,
        output_path: str,
        format: str = "wav"
    ) -> bool:
        """异步生成语音（使用轮询负载均衡）"""
        instance = self.get_next_instance()

        if not instance:
            print("没有可用的Fish Speech实例")
            return False

        url = f"http://localhost:{instance.port}/v1/tts"
        cmd = [
            '/disk0/repo/manju/fish-speech/.venv/bin/python',
            'tools/api_client.py',
            '-u', url,
            '-t', text,
            '--reference_id', reference_id,
            '-o', output_path,
            '--format', format,
            '--no-play'
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd='/disk0/repo/manju/fish-speech',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0:
                return True
            else:
                print(f"生成失败: {stderr.decode('utf-8')}")
                return False
        except asyncio.TimeoutError:
            print("生成超时")
            return False
        except Exception as e:
            print(f"生成出错: {e}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fish Speech 多实例管理器')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 启动命令
    start_parser = subparsers.add_parser('start', help='启动实例')
    start_group = start_parser.add_mutually_exclusive_group(required=True)
    start_group.add_argument('--single', type=int, metavar='GPU_ID', help='启动单GPU实例')
    start_group.add_argument('--multi', type=int, nargs='?', const=None, metavar='NUM_GPUS',
                             help='启动多GPU实例（可选指定GPU数量）')

    # 停止命令
    stop_parser = subparsers.add_parser('stop', help='停止实例')
    stop_parser.add_argument('--instance', type=int, help='停止指定实例')
    stop_parser.add_argument('--all', action='store_true', help='停止所有实例')

    # 状态命令
    subparsers.add_parser('status', help='显示实例状态')

    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试语音生成')
    test_parser.add_argument('-t', '--text', required=True, help='要生成的文本')
    test_parser.add_argument('-r', '--reference_id', required=True, help='音色ID')
    test_parser.add_argument('-o', '--output', required=True, help='输出路径')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = FishSpeechManager()

    if args.command == 'start':
        if args.single is not None:
            manager.start_single_gpu(args.single)
        elif args.multi is not None:
            manager.start_multi_gpu(args.multi)
        else:
            manager.start_multi_gpu()

        manager.status()

    elif args.command == 'stop':
        if args.all:
            manager.stop_all()
        elif args.instance:
            manager.stop_instance(args.instance)
        else:
            print("请指定 --instance 或 --all")

    elif args.command == 'status':
        manager.status()

    elif args.command == 'test':
        running_instances = [inst for inst in manager.instances.values() if inst.status == "running"]
        if not running_instances:
            print("没有运行中的实例")
            return

        client = FishSpeechClient(running_instances)
        instance = client.get_next_instance()
        if instance:
            print(f"使用实例 {instance.instance_id} (GPU {instance.gpu_id}, 端口 {instance.port})")

        cmd = [
            '/disk0/repo/manju/fish-speech/.venv/bin/python',
            'tools/api_client.py',
            '-u', f'http://localhost:{instance.port}/v1/tts',
            '-t', args.text,
            '--reference_id', args.reference_id,
            '-o', args.output,
            '--format', 'wav',
            '--no-play'
        ]
        subprocess.run(cmd, cwd='/disk0/repo/manju/fish-speech')


if __name__ == '__main__':
    main()
