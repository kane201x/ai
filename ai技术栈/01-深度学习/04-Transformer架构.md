# Transformer 架构

## 1. 原始 Transformer

### 整体结构

```mermaid
graph TD
    subgraph Encoder xN
    E_IN[输入嵌入] --> E_PE[位置编码]
    E_PE --> E_MHA[多头自注意力]
    E_MHA --> E_ADD1[残差连接+LayerNorm]
    E_ADD1 --> E_FFN[前馈网络 FFN]
    E_FFN --> E_ADD2[残差连接+LayerNorm]
    E_ADD2 --> E_OUT[Encoder输出]
    end
    subgraph Decoder xN
    D_IN[输出嵌入] --> D_PE[位置编码]
    D_PE --> D_MSK[掩码多头自注意力]
    D_MSK --> D_ADD1[残差连接+LayerNorm]
    D_ADD1 --> D_CA[交叉注意力]
    E_OUT --> D_CA
    D_CA --> D_ADD2[残差连接+LayerNorm]
    D_ADD2 --> D_FFN[前馈网络 FFN]
    D_FFN --> D_ADD3[残差连接+LayerNorm]
    D_ADD3 --> D_LIN[线性投影]
    D_LIN --> D_SOFT[Softmax]
    end
    style E_MHA fill:#4a9,color:#fff
    style D_CA fill:#f90,color:#fff
    style D_MSK fill:#f66,color:#fff
```

### 缩放点积注意力 Scaled Dot-Product Attention

```mermaid
graph TD
    Q[Q: b×h×n×d_k] --> MUL[Q × Kᵀ]
    K[K: b×h×n×d_k] --> MUL
    MUL --> SCALE[÷ √d_k]
    SCALE --> MASK[(可选掩码)]
    MASK --> SOFT[Softmax]
    SOFT --> ATT[Attention × V]
    V[V: b×h×n×d_v] --> ATT
    ATT --> OUT[输出: b×h×n×d_v]
    style SOFT fill:#f96,color:#fff
    style SCALE fill:#69f,color:#fff
```

- **公式**：Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Q**：Query 查询，当前词要"问"什么
- **K**：Key 键，所有词"被问"的回答特征
- **V**：Value 值，实际要提取的信息
- **缩放因子 1/√d_k**：防止点积过大使 softmax 梯度消失

手动实现注意力机制：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn

q = torch.randn(4, 8, 16, 64)
k = torch.randn(4, 8, 16, 64)
v = torch.randn(4, 8, 16, 64)
out, attn_weights = scaled_dot_product_attention(q, k, v)
```

### 多头注意力 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        b = Q.size(0)
        Q = self.W_q(Q).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(b, -1, self.n_heads * self.d_k)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(4, 16, 512)
out = mha(x, x, x)  # Self-attention
```

### 位置编码

| 方法 | 公式/原理 | 使用模型 |
|------|-----------|---------|
| Sinusoidal PE | PE(pos,2i)=sin(pos/10000^{2i/d}) | 原始 Transformer |
| 可学习 PE | 嵌入表：nn.Embedding(max_len, d_model) | BERT |
| RoPE | 旋转矩阵编码相对位置 | LLaMA, GPT-NeoX |
| ALiBi | 线性偏置注意力分数 | BLOOM |
| T5 Bias | 可学习相对位置偏置 | T5 |
| NoPE | 无位置编码（因果模型隐含位置） | 部分实验 |

### 前馈网络 FFN

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation=F.relu, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = activation

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

# SwiGLU FFN (LLaMA 系列)
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        d_ff = int(2 * d_ff / 3)  # SwiGLU 有3个投影，调整维度
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_ff)
        self.fc3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))
```

### 完整 Transformer Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

## 2. Transformer 为什么有效

| 特性 | CNN | RNN/LSTM | Transformer |
|------|-----|----------|------------|
| 长程依赖 | 需深层堆叠 | 梯度消失问题 | ✓ 直接 O(1) 路径 |
| 并行计算 | ✓ | ✗ 串行 | ✓ |
| 全局感受野 | ✗ 局部 | ✓ | ✓ |
| 计算复杂度 | O(n·k²·c) | O(n·d²) | O(n²·d) |
| 参数量 | 少 | 中 | 大 |
| 可解释性 | 特征图 | 隐状态 | 注意力权重 |
| 位置编码 | 隐式(空间结构) | 隐式(时间步) | 显式需要 |

## 3. 关键改进

### Pre-LN vs Post-LN

```mermaid
graph LR
    subgraph "Post-LN (原始)"
        A[输入x] --> LN[LayerNorm] 
        LN --> ATTN[Attention] 
        ATTN --> ADD[残差相加：x + ATTN] 
        ADD --> OUT[输出]
    end
    subgraph "Pre-LN (现代)"
        B[输入x] --> ATTN2[Attention] 
        ATTN2 --> ADD2[残差相加：x + ATTN2] 
        ADD2 --> LN2[LayerNorm] 
        LN2 --> OUT2[输出]
    end
    style LN fill:#f96
    style LN2 fill:#4a9
```

- **Post-LN（原始）**：残差之前做 LN，训练不稳定，需 warmup
- **Pre-LN（现代）**：残差之后做 LN，训练更稳定，无需 warmup
- 所有现代模型（GPT/BERT/LLaMA/T5）都使用 Pre-LN

### 注意力变体

```python
# Flash Attention V2 使用方式
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v)

# Grouped Query Attention (GQA)
class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k * n_kv_heads)
        self.W_v = nn.Linear(d_model, self.d_k * n_kv_heads)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        b = Q.size(0)
        Q = self.W_q(Q).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(b, -1, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(b, -1, self.n_kv_heads, self.d_k).transpose(1, 2)
        K = K.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        V = V.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        return F.scaled_dot_product_attention(Q, K, V, mask)
```

- **Flash Attention V1-V3**：IO 感知，分块计算，减少 HBM 读写
- **PagedAttention**：类虚拟内存 KV Cache 管理
- **Multi-Query / Grouped Query Attention**：减少 KV 头数
- **Multi-head Latent Attention**：低秩压缩 KV Cache（DeepSeek）

### 高效 Transformer

| 方法 | 复杂度 | 核心思想 | 适用 |
|------|--------|---------|------|
| Linformer | O(n) | 低秩近似 KV | 中等长度 |
| Performer | O(n) | FAVOR+ 核方法 | 通用 |
| Reformer | O(n log n) | LSH 哈希 + 可逆层 | 超长序列 |
| Longformer | O(n) | 滑动窗口+全局 | 文档 |
| BigBird | O(n) | 稀疏+全局+随机 | 通用 |
| FlashAttention | O(n²)实际快 | IO 感知 CUDA 优化 | GPU |

## 4. 模型维度配置

| 模型 | d_model | Layers | Heads | d_ff | 参数量 |
|------|---------|--------|-------|------|--------|
| BERT-Base | 768 | 12 | 12 | 3072 | 110M |
| BERT-Large | 1024 | 24 | 16 | 4096 | 340M |
| GPT-2 XL | 1600 | 48 | 25 | 6400 | 1.5B |
| GPT-3 | 12288 | 96 | 96 | 49152 | 175B |
| LLaMA 7B | 4096 | 32 | 32 | 11008 | 6.7B |
| LLaMA 65B | 8192 | 80 | 64 | 22016 | 65B |
| DeepSeek V2 | 4096 | 60 | 32 | 12288 | 236B(MoE) |
| DeepSeek V4 | 7168 | 78 | 56 | 21504 | ~1T(MoE) |

## 5. Transformer 变体

### Encoder-Only（BERT 类）
- 双向注意力，适合理解任务
- 分类、NER、抽取式 QA

### Decoder-Only（GPT 类）
- 因果注意力（只看到左侧）
- 生成任务、LLM、对话

### Encoder-Decoder（T5 类）
- 完整 Seq2Seq 结构
- 翻译、摘要、多任务统一

```python
# GPT 风格因果自回归模型
class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=~mask)
        return self.head(self.ln(x))
```

## 6. 2025-2026 趋势
- **混合注意力**：Transformer + SSM（Mamba）混合层
- **超大 MoE Transformer**：DeepSeek 等将 MoE 路由引入 Transformer
- **线性注意力**：在不牺牲质量下接近 O(n) 复杂度
- **长上下文优化**：百万级 token 成为现实
- **Speculative Decoding**：2-3× 推理加速
- **KV Cache 量化**：INT4/FP8 缓存压缩

## 7. 案例：从零训练一个微型 GPT（因果 LM 实战）

演示 mini-GPT 的构建、因果掩码与自回归训练一步，可直接扩展到真实语料。

```mermaid
graph LR
    A["token序列"] --> B["Embedding+位置"]
    B --> C["N× Pre-LN Block\n(MHA+SwiGLU)"]
    C --> D["LayerNorm"]
    D --> E["LM Head\n→词表"]
    E --> F["Shifted CrossEntropy"]
    F --> G["自回归训练"]
    style C fill:#4a9,color:#fff
    style E fill:#f96,color:#fff
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab: int = 100, d: int = 128, n_layers: int = 4, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1024, d))
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d, heads, d * 4) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        t = x.size(1)
        x = self.emb(x) + self.pos[:t]
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        for blk in self.blocks:
            x = blk(x, mask=~mask)
        return self.head(self.ln(x))               # 形状: [b, t, vocab]

vocab, b, t = 100, 4, 20
model = MiniGPT(vocab)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
x = torch.randint(0, vocab, (b, t))
logits = model(x)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
opt.zero_grad(); loss.backward(); opt.step()
print("GPT 训练 loss:", round(loss.item(), 3))
```

## 8. 案例：位置编码方案对比与 RoPE 实现

不同位置编码在「外推性 / 相对位置感知」上差异显著，现代 LLM 主流用 RoPE。

| 编码 | 相对位置 | 长度外推 | 实现成本 | 采用模型 |
|------|---------|---------|---------|---------|
| Sinusoidal | ✓(三角性质) | 中 | 低 | 原始 Transformer |
| 可学习 PE | ✗ | 差(固定长度) | 低 | BERT |
| RoPE | ✓(旋转) | 好 | 中 | LLaMA/Qwen |
| ALiBi | ✓(线性偏置) | 好 | 低 | BLOOM |

```python
def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10000):
    """q,k: [b, h, n, d]，d 为偶数。返回旋转后 q,k。"""
    b, h, n, d = q.shape
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    pos = torch.arange(n).float()
    freqs = torch.outer(pos, inv_freq)            # [n, half]
    cos = freqs.cos().repeat_interleave(2, -1)    # [n, d]
    sin = freqs.sin().repeat_interleave(2, -1)
    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        rot_part = torch.cat([-x2, x1], -1)
        return x * cos + rot_part * sin
    return rot(q), rot(k)                          # 形状不变

q = torch.randn(2, 4, 16, 64)
k = torch.randn(2, 4, 16, 64)
q_r, k_r = apply_rope(q, k)
print("RoPE 后形状:", tuple(q_r.shape))            # (2, 4, 16, 64)
```
