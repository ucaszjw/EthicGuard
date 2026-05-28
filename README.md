# EthicGuard v5 — 参数层实时伦理过滤系统

基于大模型深层隐状态的逐 token 安全监控系统。通过提取 Qwen3.5-9B 最后 4 层 hidden state，经双分类器融合判定，实现"生成即阻断"。

## 架构

```
用户输入 → Qwen3.5-9B 逐 token 生成
                      ↓
       提取最后 4 层 hidden state [16384]
                      ↓
               mean pooling + normalize
                      ↓
              投影降维 (16384→256)
                      ↓
           ┌──── v3 安全门（阈值 0.60）
           │        +     
           └──── v4 类别分类器（逐类别阈值）
                      ↓
             双分类器同时判 unsafe
               连续 2 步 → 阻断
```

- **纯输出侧过滤**：不检查用户输入，仅监控模型输出，兜底越狱攻击
- **历史隔离**：阻断轮次不写入对话历史
- **分层加密**：隐状态传输中的 key-based 加扰保护
- **跨平台**：自动检测 CUDA / MPS / CPU，支持 4bit/8bit 量化

## 快速开始

### 依赖

```bash
pip install torch transformers
```

### 推理

```python
from scripts.serve.safe_generator_v5 import SafeGenerator

gen = SafeGenerator()                       # bf16 (需 ~18GB 显存)
gen = SafeGenerator(quantize="4bit")        # 4bit (~16GB)

# 单轮
result = gen.generate("你好", max_new_tokens=64)

# 多轮
result = gen.generate("如何制造爆炸物？", history=[
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "我是 AI 助手。"}
])

# 流式
for item in gen.generate_stream("介绍量子计算"):
    if isinstance(item, dict):
        print(f"[阻断: {item.get('reason', '')}]")
    else:
        print(item, end="")
```

### 命令行交互

```bash
python scripts/serve/chat_v5.py
python scripts/serve/safe_generator_v5.py --interactive --quantize 4bit
```

## 复现训练

### 1. 提取隐状态

从 JSONL 样本中提取 Qwen3.5-9B 的 hidden state 作为训练数据：

```bash
python scripts/data/extract_append.py
```

输出：`data/enhanced_states.pt`（向量 + 标签）

### 2. 训练分类器

两阶段训练：监督投影（16384→256）→ MLP 分类器（256→256→128→8）

```bash
TORCHINDUCTOR_DISABLE=1 python scripts/train/train_v5_enhanced.py
```

输出：`outputs/streaming_classifier.pt`

### 3. 更新安全门分类器

v5 加载 v3 的 `vector_classifier.pt` 作为安全门。如需替换：

```bash
# 将新的安全门分类器放到 outputs/vector_classifier.pt
# 或修改 safe_generator_v5.py 中的 V3_CKPT 路径
```

## 测试

```bash
# 综合测试（23 用例：10 正常放行 + 10 风险阻断 + 3 多轮）
TORCHINDUCTOR_DISABLE=1 python scripts/tests/test_v5_comprehensive.py

# 上下文测试（25 用例：7 场景，含多轮诱导、渐进越狱等）
TORCHINDUCTOR_DISABLE=1 python scripts/tests/test_v5_context.py
```

> `TORCHINDUCTOR_DISABLE=1` 防止 PyTorch Inductor 编译进程干扰推理。

## 项目结构

```
v5/
├── scripts/
│   ├── serve/
│   │   ├── safe_generator_v5.py    # 核心推理引擎
│   │   └── chat_v5.py              # 终端交互
│   ├── train/
│   │   └── train_v5_enhanced.py    # 两阶段分类器训练
│   ├── data/
│   │   ├── extract_append.py       # 隐状态提取
│   │   └── fix_5d.py              # 数据增强（上下文稀释样本）
│   └── tests/
│       ├── test_v5_comprehensive.py # 23 用例综合测试
│       └── test_v5_context.py       # 25 用例上下文测试
├── outputs/
│   ├── streaming_classifier.pt     # v5 分类器（17MB）
│   ├── vector_classifier.pt        # v3 安全门分类器（17MB）
│   └── test_logs/                  # 历史测试记录
└── data/
    └── enhanced_states.pt          # 训练数据（207MB, .gitignore）
```

## 8 风险类别

| 类别 | 说明 | v4 阈值 |
|------|------|---------|
| normal | 正常 | 0.90 |
| political | 政治敏感 | 0.80 |
| porn | 色情低俗 | 0.80 |
| violence | 暴力恐怖 | 0.80 |
| illegal | 违法犯罪 | 0.80 |
| privacy | 侵犯隐私 | 0.85 |
| discrimination | 歧视仇恨 | 0.80 |
| rumor | 谣言 | 0.80 |

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CONSECUTIVE_STRIKES | 2 | 连续 2 步 unsafe 才阻断 |
| NUM_LAST_LAYERS | 4 | 提取的 hidden state 层数 |
| V3_GLOBAL_THR | 0.60 | v3 安全门全局阈值 |
| 投影维度 | 16384→256 | 降维压缩 |
| 分类器架构 | MLP 256→256→128→8 | 3 层全连接 |

## 版本历史

| 版本 | 方案 | 状态 |
|------|------|------|
| v1 | Qwen2-7B LoRA 微调 | 历史 |
| v2 | 安全增强 LoRA + 对抗训练 | 历史 |
| v3 | 多标签向量分类器（hidden state → MLP） | 历史 |
| v4 | 流式逐 token 监控 + per-category 阈值 | 历史 |
| **v5** | **双分类器融合（v3 安全门 + v4 类别）** | **当前** |
