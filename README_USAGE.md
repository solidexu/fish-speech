# Fish Speech 多实例并行生成使用指南

本文档说明如何使用 Fish Speech 的多实例并行生成功能，提高语音生成吞吐量。

## 目录

- [概述](#概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [多实例管理](#多实例管理)
- [并行生成](#并行生成)
- [性能测试](#性能测试)
- [常见问题](#常见问题)

---

## 概述

### 什么是多实例并行生成？

Fish Speech 多实例并行生成允许在多个 GPU 上同时运行多个 TTS 实例，通过负载均衡策略实现：

- ✅ **吞吐量提升**：N 个实例可提供 N 倍吞吐量
- ✅ **GPU 利用率最大化**：充分利用多 GPU 资源
- ✅ **故障隔离**：单个实例故障不影响其他实例
- ✅ **负载均衡**：轮询策略确保任务均匀分配

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 多实例管理器 | `fish_speech_manager.py` | 启动/停止/监控多个实例 |
| 同步启动器 | `sync_start.py` | 同时启动多个实例 |
| 音色管理 | `add_voice.py` / `add_voice.sh` | 添加和管理音色库 |
| 控制脚本 | `fish_speech_control.sh` | 快捷控制命令 |
| 音色库构建 | `tools/build_voice_lib.py` | 构建音色参考库 |

---

## 安装

### 1. 系统要求

- **Python**: 3.10+
- **GPU**: NVIDIA GPU，每个实例需要 ≥23GB 显存
- **CUDA**: 11.8+ 或 12.1+
- **操作系统**: Linux (推荐 Ubuntu 20.04+)

### 2. 安装依赖

#### 方法 A：使用 uv（推荐，快速）

```bash
cd /disk0/repo/manju/third_party/fish-speech

# 安装 uv
pip install uv

# 同步依赖（自动创建虚拟环境）
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

#### 方法 B：使用 pip

```bash
cd /disk0/repo/manju/third_party/fish-speech

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .

# 或安装核心依赖（更快）
pip install torch==2.8.0 torchaudio==2.8.0 transformers<=4.57.3 \
    lightning hydra-core librosa gradio uvicorn loguru \
    pyaudio ormsgpack pydantic==2.9.2 descript-audio-codec safetensors
```

### 3. 下载模型

```bash
# 创建 checkpoints 目录
mkdir -p checkpoints/s2-pro

# 下载 S2-Pro 模型（约 4GB）
# 方式1：从 HuggingFace 下载
# https://huggingface.co/fishaudio/s2-pro

# 方式2：使用 modelscope
pip install modelscope
modelscope download --model fishaudio/s2-pro --local_dir checkpoints/s2-pro

# 验证模型文件
ls -lh checkpoints/s2-pro/
# 应包含：
# - 模型文件（如 model.safetensors 或 pytorch_model.bin）
# - codec.pth（解码器）
# - config.json
```

---

## 快速开始

### 1. 启动单实例

```bash
cd /disk0/repo/manju/third_party/fish-speech

# 启动单 GPU 实例（GPU 0）
python fish_speech_manager.py start --single 0

# 查看状态
python fish_speech_manager.py status
```

**预期输出：**
```
Fish Speech 实例状态:
----------------------------------------------------------------------
ID   GPU  端口     PID      状态
----------------------------------------------------------------------
1    0    9997   12345    running
----------------------------------------------------------------------
```

### 2. 测试生成

```bash
# 使用 API 客户端测试
python tools/api_client.py \
    -u http://localhost:9997/v1/tts \
    -t "这是一个测试语音" \
    --reference_id paimon \
    -o test_output.wav \
    --format wav \
    --no-play

# 检查生成的文件
ls -lh test_output.wav
```

---

## 多实例管理

### 1. 启动多 GPU 实例

#### 方法 A：自动检测可用 GPU

```bash
# 启动所有可用 GPU（自动检测显存≥23GB的GPU）
python fish_speech_manager.py start --multi
```

#### 方法 B：指定 GPU 数量

```bash
# 启动 2 个实例
python fish_speech_manager.py start --multi 2
```

#### 方法 C：使用同步启动器

```bash
# 同时启动 GPU 0 和 GPU 1
python sync_start.py start --gpus 0,1

# 查看状态
python sync_start.py status
```

### 2. 实例状态管理

```bash
# 查看所有实例状态
python fish_speech_manager.py status

# 停止指定实例
python fish_speech_manager.py stop --instance 1

# 停止所有实例
python fish_speech_manager.py stop --all

# 重启实例
python fish_speech_manager.py stop --instance 1
python fish_speech_manager.py start --single 0
```

### 3. 实例配置

配置文件：`multi_instance_config.json`

```json
{
  "instances": [
    {
      "instance_id": 1,
      "gpu_id": 0,
      "port": 9997,
      "pid": 12345,
      "status": "running"
    },
    {
      "instance_id": 2,
      "gpu_id": 1,
      "port": 9998,
      "pid": 12346,
      "status": "running"
    }
  ]
}
```

**配置说明：**
- `instance_id`: 实例唯一标识
- `gpu_id`: 使用的 GPU 编号
- `port`: API 服务端口（9997, 9998, 9999...）
- `pid`: 进程 ID
- `status`: 实例状态（stopped/starting/running/error）

### 4. GPU 资源监控

```bash
# 查看 GPU 状态
nvidia-smi

# 查看可用 GPU（显存≥23GB）
python fish_speech_manager.py status
# 输出包含：可用GPU: [0, 1]
```

---

## 并行生成

### 1. 使用 Python API

```python
import asyncio
from fish_speech_manager import FishSpeechManager, FishSpeechClient

async def parallel_generation():
    # 初始化管理器
    manager = FishSpeechManager()
    manager.status()
    
    # 获取运行中的实例
    running_instances = [
        inst for inst in manager.instances.values() 
        if inst.status == "running"
    ]
    
    if not running_instances:
        print("没有运行中的实例")
        return
    
    # 创建客户端（自动负载均衡）
    client = FishSpeechClient(running_instances)
    
    # 并行生成多个语音
    tasks = []
    texts = [
        "这是第一段语音",
        "这是第二段语音",
        "这是第三段语音",
        "这是第四段语音",
        "这是第五段语音"
    ]
    
    for i, text in enumerate(texts):
        task = client.generate_speech_async(
            text=text,
            reference_id="paimon",  # 音色 ID
            output_path=f"/tmp/output_{i+1}.wav",
            format="wav"
        )
        tasks.append(task)
    
    # 并行执行
    results = await asyncio.gather(*tasks)
    
    # 统计结果
    success_count = sum(1 for r in results if r)
    print(f"成功生成 {success_count}/{len(tasks)} 个语音")

# 运行
asyncio.run(parallel_generation())
```

### 2. 使用 HTTP API（推荐）

```python
import asyncio
import aiohttp

async def generate_speech(session, port, text, reference_id, output_path):
    """直接通过 HTTP API 生成"""
    url = f"http://localhost:{port}/v1/tts"
    
    payload = {
        "text": text,
        "reference_id": reference_id,
        "format": "wav"
    }
    
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            audio_data = await response.read()
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            return True
        return False

async def parallel_generation_http():
    """使用 HTTP API 并行生成"""
    
    # 实例端口列表
    ports = [9997, 9998]  # 2个实例
    
    texts = [
        "文本1", "文本2", "文本3", "文本4", "文本5",
        "文本6", "文本7", "文本8", "文本9", "文本10"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, text in enumerate(texts):
            # 轮询选择实例
            port = ports[i % len(ports)]
            
            task = generate_speech(
                session, port, text, "paimon", 
                f"/tmp/output_{i+1}.wav"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r)
        print(f"成功: {success_count}/{len(tasks)}")

asyncio.run(parallel_generation_http())
```

### 3. 负载均衡策略

Fish Speech Client 使用**轮询（Round-Robin）**策略：

```python
class FishSpeechClient:
    def get_next_instance(self):
        """轮询选择下一个实例"""
        running_instances = [
            inst for inst in self.instances 
            if inst.status == "running"
        ]
        
        if not running_instances:
            return None
        
        # 轮询选择
        instance = running_instances[self.current_idx % len(running_instances)]
        self.current_idx += 1
        
        return instance
```

**优势：**
- 任务均匀分配到所有实例
- 避免单个实例过载
- 简单高效，无状态管理

---

## 性能测试

### 1. 运行性能测试

```bash
cd /disk0/repo/manju/third_party/fish-speech

# 测试 10 个任务并行生成
python test_parallel_api.py -n 10

# 测试 50 个任务
python test_parallel_api.py -n 50

# 测试 100 个任务
python test_parallel_api.py -n 100
```

### 2. 性能指标

**预期性能（2 个实例）：**

| 任务数 | 单实例耗时 | 双实例并行耗时 | 加速比 |
|--------|-----------|--------------|--------|
| 10 | ~50s | ~25s | 2x |
| 50 | ~250s | ~125s | 2x |
| 100 | ~500s | ~250s | 2x |

**吞吐量对比：**
- 单实例：~0.2 任务/秒
- 双实例：~0.4 任务/秒（2x）
- 四实例：~0.8 任务/秒（4x）

### 3. 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看每个实例的 GPU 占用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 音色管理

### 1. 添加音色

```bash
# 使用 Python 脚本
python add_voice.py \
    --name "新音色" \
    --reference_audio "参考音频.wav" \
    --reference_text "参考文本"

# 使用 Shell 脚本
./add_voice.sh "新音色" "参考音频.wav" "参考文本"
```

### 2. 音色库结构

```
checkpoints/voices/
├── paimon/
│   ├── reference.wav      # 参考音频
│   ├── reference.txt      # 参考文本
│   └── metadata.json      # 音色元数据
├── calm_male/
│   ├── reference.wav
│   ├── reference.txt
│   └── metadata.json
└── sweet_female/
    ├── reference.wav
    ├── reference.txt
    └── metadata.json
```

### 3. 使用音色

```bash
# 指定音色 ID
python tools/api_client.py \
    -u http://localhost:9997/v1/tts \
    -t "测试文本" \
    --reference_id paimon \
    -o output.wav
```

---

## 常见问题

### Q1: 实例启动失败，提示 "ModuleNotFoundError: No module named 'torch'"

**原因：** 未安装依赖

**解决方案：**
```bash
cd /disk0/repo/manju/third_party/fish-speech
uv sync  # 或 pip install -e .
```

### Q2: 实例启动超时

**原因：** 模型加载耗时较长（约 30-60 秒）

**解决方案：**
- 增加启动超时时间（默认 60 秒）
- 检查 GPU 显存是否足够（需要 ≥23GB）
- 查看启动日志：`/tmp/fish_speech_instances/instance_*.log`

### Q3: GPU 显存不足

**原因：** 模型占用约 22GB，需要额外 buffer

**解决方案：**
```bash
# 检查 GPU 显存
nvidia-smi --query-gpu=index,memory.free --format=csv

# 只使用显存≥23GB的GPU
python fish_speech_manager.py start --multi
```

### Q4: 无法连接到 API 端口

**原因：** 实例未真正启动或端口被占用

**解决方案：**
```bash
# 检查端口监听
netstat -tlnp | grep 9997

# 检查实例进程
ps aux | grep api_server

# 重启实例
python fish_speech_manager.py stop --all
python fish_speech_manager.py start --multi
```

### Q5: 并行生成性能不佳

**原因：** 任务分配不均或实例数量不足

**解决方案：**
- 确保所有实例都在运行：`python fish_speech_manager.py status`
- 增加 GPU 实例数量
- 检查网络连接是否稳定
- 使用 HTTP API 而非命令行客户端

### Q6: 如何在 Manju 项目中使用？

**集成示例：**

```python
# 在 manju/services/tts_client.py 中使用
from manju.third_party.fish_speech.fish_speech_manager import FishSpeechClient

class TTSClient:
    def __init__(self, config):
        self.fish_speech_client = FishSpeechClient([
            FishSpeechInstance(instance_id=1, gpu_id=0, port=9997),
            FishSpeechInstance(instance_id=2, gpu_id=1, port=9998)
        ])
    
    async def generate(self, text, voice_id, output_path):
        return await self.fish_speech_client.generate_speech_async(
            text=text,
            reference_id=voice_id,
            output_path=output_path
        )
```

---

## 进阶使用

### 1. 自定义启动参数

```bash
# 修改 fish_speech_manager.py 中的启动命令
cmd = [
    'python3',
    'tools/api_server.py',
    '--llama-checkpoint-path', 'checkpoints/s2-pro',
    '--decoder-checkpoint-path', 'checkpoints/s2-pro/codec.pth',
    '--listen', f'0.0.0.0:{port}',
    '--half',  # 使用 FP16（节省显存）
    '--max-length', 2048,  # 最大生成长度
    '--compile',  # 使用 torch.compile 加速
]
```

### 2. 分布式部署

**场景：** 多台服务器协同工作

```python
# 客户端连接远程实例
instances = [
    FishSpeechInstance(instance_id=1, gpu_id=0, port=9997),  # 本地
    FishSpeechInstance(instance_id=2, gpu_id=0, port=9997),  # 远程服务器1
    FishSpeechInstance(instance_id=3, gpu_id=0, port=9997),  # 远程服务器2
]

# 需要修改 API URL
url = f"http://remote-server1:{instance.port}/v1/tts"
```

### 3. 动态扩缩容

```python
# 根据负载动态增减实例
manager = FishSpeechManager()

# 检测高负载，启动新实例
if queue_length > threshold:
    manager.start_instance(
        instance_id=new_id,
        gpu_id=available_gpu,
        port=9997 + new_id
    )

# 检测低负载，停止实例
if queue_length < threshold and len(running_instances) > 1:
    manager.stop_instance(instance_id=last_instance)
```

---

## 日志和监控

### 1. 查看实例日志

```bash
# 实例启动日志
tail -f /tmp/fish_speech_instances/instance_1_gpu0.log

# 实例运行日志
tail -f /tmp/fish_speech_instances/instance_2_gpu1.log
```

### 2. API 日志

```bash
# 启动 API 服务时添加日志级别
python tools/api_server.py \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --listen 0.0.0.0:9997 \
    --log-level DEBUG
```

### 3. 性能监控

```bash
# GPU 使用监控
nvidia-smi dmon -s u

# 进程监控
htop -p <pid>

# 网络监控
iftop -i eth0
```

---

## 相关文档

- [Fish Speech 官方文档](https://speech.fish.audio/)
- [API 服务文档](https://speech.fish.audio/server/)
- [命令行推理](https://speech.fish.audio/inference/)
- [模型下载](https://huggingface.co/fishaudio/s2-pro)
- [技术报告](https://arxiv.org/abs/2603.08823)

---

## 总结

Fish Speech 多实例并行生成功能提供了：

✅ **高性能**：N 倍吞吐量提升  
✅ **易用性**：自动化实例管理和负载均衡  
✅ **可靠性**：实例独立运行，故障隔离  
✅ **扩展性**：支持动态扩缩容和分布式部署  

**推荐配置：**
- 2 个 GPU：吞吐量提升 2x
- 4 个 GPU：吞吐量提升 4x
- 8 个 GPU：吞吐量提升 8x

**开始使用：**
```bash
# 1. 安装依赖
uv sync

# 2. 启动多实例
python fish_speech_manager.py start --multi

# 3. 并行生成
python test_parallel_api.py -n 10
```

**祝你使用愉快！** 🎉