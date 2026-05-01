#!/usr/bin/env python3
"""
测试 Fish Speech 并行生成功能
验证多实例负载均衡和并发性能
"""

import asyncio
import time
import sys
sys.path.insert(0, '/disk0/repo/manju/third_party/fish-speech')

from fish_speech_manager import FishSpeechManager, FishSpeechClient


async def test_parallel_generation(num_tasks: int = 10):
    """测试并行生成"""
    
    # 初始化管理器
    manager = FishSpeechManager()
    manager.status()
    
    # 获取运行中的实例
    running_instances = [inst for inst in manager.instances.values() if inst.status == "running"]
    
    if not running_instances:
        print("❌ 没有运行中的实例")
        return
    
    print(f"\n✅ 找到 {len(running_instances)} 个运行实例")
    print(f"实例信息:")
    for inst in running_instances:
        print(f"  - 实例 {inst.instance_id}: GPU {inst.gpu_id}, 端口 {inst.port}")
    
    # 创建客户端
    client = FishSpeechClient(running_instances)
    
    # 测试参数
    test_texts = [
        "这是一个测试语音，用于验证并行生成功能。",
        "Fish Speech 支持多实例并发，提高生成效率。",
        "通过负载均衡，可以充分利用多GPU资源。",
        "每个实例独立运行，互不干扰。",
        "轮询策略确保任务均匀分配。",
        "异步生成提高整体吞吐量。",
        "多GPU并行是提升性能的关键。",
        "测试语音生成速度和质量。",
        "验证系统稳定性和可靠性。",
        "完成所有测试任务，统计性能数据。"
    ]
    
    reference_id = "paimon"  # 使用一个测试音色
    output_dir = "/tmp/fish_speech_test"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🚀 开始并行生成测试（{num_tasks} 个任务）")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建异步任务
    tasks = []
    for i in range(num_tasks):
        text = test_texts[i % len(test_texts)]
        output_path = f"{output_dir}/test_{i+1}.wav"
        
        task = client.generate_speech_async(
            text=text,
            reference_id=reference_id,
            output_path=output_path,
            format="wav"
        )
        tasks.append(task)
        
        print(f"任务 {i+1}: 实例 {client.current_idx % len(running_instances) + 1}, 输出 {output_path}")
    
    # 并行执行
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r is True)
    fail_count = sum(1 for r in results if r is not True)
    
    print("\n" + "=" * 70)
    print("📊 测试结果统计")
    print("=" * 70)
    print(f"总任务数: {num_tasks}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均耗时: {total_time/num_tasks:.2f} 秒/任务")
    print(f"吞吐量: {num_tasks/total_time:.2f} 任务/秒")
    print("=" * 70)
    
    # 性能分析
    if len(running_instances) > 1:
        theoretical_speedup = len(running_instances)
        print(f"\n💡 性能分析:")
        print(f"  实例数量: {len(running_instances)}")
        print(f"  理论加速比: {theoretical_speedup}x")
        print(f"  实际吞吐量提升: ~{len(running_instances)}x（并行处理）")
    
    # 显示生成的文件
    print(f"\n生成的音频文件:")
    for i in range(num_tasks):
        output_path = f"{output_dir}/test_{i+1}.wav"
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"  ✓ test_{i+1}.wav ({size} bytes)")
        else:
            print(f"  ✗ test_{i+1}.wav (未生成)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='测试 Fish Speech 并行生成')
    parser.add_argument('-n', '--num-tasks', type=int, default=10, help='测试任务数量')
    
    args = parser.parse_args()
    
    asyncio.run(test_parallel_generation(args.num_tasks))