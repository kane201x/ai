# LLMOps

## 1. 定义与范围
LLMOps = LLM + MLOps，专门针对大语言模型的工程化运维。

### 核心挑战

| 挑战 | 具体表现 | 影响 | 解决方案 |
|------|---------|------|---------|
| 成本 | API / GPU 费用高，Token 消耗大 | 月账单 $10K+ | Prompt 压缩、缓存、模型路由 |
| 质量 | 幻觉、偏见、安全性、一致性 | 用户信任下降 | Guardrails、RAG、人类反馈 |
| 监控 | 延迟、错误率、用户满意度 | 用户体验差 | 全链路追踪、告警 |
| 迭代 | Prompt 版本混乱、A/B 测试困难 | 改进效率低 | Prompt Registry、版本管理 |
| 安全 | 提示注入、数据泄露 | 合规风险 | 输入/输出护栏、PII 检测 |

### LLMOps 与 MLOps 对比

| 维度 | MLOps | LLMOps |
|------|-------|--------|
| 模型 | 训练自己的模型 (CNN/RNN/Transformer) | 使用/微调预训练 LLM |
| 数据 | 结构化/半结构化特征 | 非结构化文本 + Prompt |
| 部署 | 模型服务器 + 批处理推理 | LLM 推理引擎 (vLLM/TGI) |
| 监控 | 数据漂移、概念漂移、特征分布 | 幻觉、忠实度、Token 消耗、安全 |
| 评估 | 准确率、F1、AUC | BLEU、ROUGE、忠实度、人工评估 |
| 成本 | 训练 GPU + 推理 1× | 推理 Token 计费 + 大 GPU 集群 |
| 迭代周期 | 周-月 (重新训练) | 分钟-天 (Prompt 更新) |

## 2. Prompt 管理

### 版本控制策略

| 层次 | 管理内容 | 版本方式 | 回滚能力 |
|------|---------|---------|---------|
| Prompt 模板 | 系统提示词、指令模板 | Git + 语义版本 (v1.0, v1.1) | ✅ GitHub |
| 参数配置 | temperature, top_p, max_tokens | 配置文件 YAML | ✅ 配置中心 |
| 示例 (Few-shot) | 上下文示例选择 | 数据库版本 | ✅ 数据库还原 |
| A/B 测试配置 | 流量分配比例 | 配置中心 | ✅ 实时切换 |

### 评估自动化

```mermaid
flowchart TD
    subgraph Prompt 开发
    A[编写 Prompt 模板] --> B[注册到 Prompt Registry]
    B --> C[分配版本号 v1.0]
    end
    subgraph 离线评估
    C --> D[测试用例集 N=500]
    D --> E[批量 LLM 调用]
    E --> F[自动评估器]
    F --> G{通过标准?}
    G -->|否| H[分析失败用例]
    H --> I[修改 Prompt]
    I --> B
    end
    subgraph 在线 A/B
    G -->|是| J[部署 v1.0 生产 5% 流量]
    J --> K[在线指标收集]
    K --> L{优于基线 v0.9?}
    L -->|否| M[回滚到 v0.9]
    L -->|是| N[扩量 50%]
    N --> O{稳定?}
    O -->|否| M
    O -->|是| P[全量 v1.0]
    end
```

### 代码示例

```python
# Prompt 版本管理 (LangChain + LangSmith)
from langchain.prompts import PromptTemplate
from langsmith import Client

client = Client()

prompt_template = PromptTemplate(
    template="""你是一个专业的客服助手。请根据以下上下文回答用户问题。

上下文：
{context}

用户问题：
{question}

请给出准确、简洁的回答。如果无法从上下文中找到答案，请明确告知。""",
    input_variables=["context", "question"],
)

client.create_prompt(
    name="customer-service-prompt",
    description="通用客服 Prompt v2.0",
    tags=["production", "customer-service"],
    prompt=prompt_template,
    metadata={
        "version": "2.0",
        "model": "gpt-4o",
        "temperature": 0.3,
        "author": "team-nlp",
    },
)

prompts = client.list_prompts(name="customer-service-prompt")
for p in prompts:
    print(f"Version: {p.metadata.get('version')} - Created: {p.created_at}")
```

```python
# RAG 评估 (RAGAS)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

questions = ["什么是向量数据库?", "如何选择嵌入模型?"]
answers = ["向量数据库是...", "选择嵌入模型需要..."]
contexts = [
    ["向量数据库是非结构化数据的向量表示存储系统..."],
    ["嵌入模型选择取决于维度、成本和任务类型..."],
]

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [
        "向量数据库用于存储和检索向量嵌入",
        "根据维度、隐私、成本选择嵌入模型",
    ],
})

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
)

print(f"Faithfulness: {result['faithfulness']:.3f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.3f}")
print(f"Context Precision: {result['context_precision']:.3f}")
print(f"Context Recall: {result['context_recall']:.3f}")
print(f"Answer Correctness: {result['answer_correctness']:.3f}")
```

```python
# 成本计算
from typing import Dict
from datetime import datetime

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "llama-3-70b": {"input": 0.85, "output": 0.85},
}

class CostTracker:
    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

    def log_call(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.calls += 1

    def calculate_cost(self) -> Dict:
        pricing = MODEL_PRICING[self.model]
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return {
            "model": self.model,
            "total_calls": self.calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
            "avg_cost_per_call": round((input_cost + output_cost) / self.calls, 6),
        }

tracker = CostTracker("gpt-4o")
tracker.log_call(input_tokens=500, output_tokens=200)
tracker.log_call(input_tokens=1200, output_tokens=600)
print(tracker.calculate_cost())

# 预算告警
if tracker.calculate_cost()["total_cost_usd"] > 100:
    print("ALERT: Monthly budget exceeded!")
```

```python
# Guardrails 实现 (NeMo Guardrails)
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("config/guardrails")

rails = LLMRails(config)

rails.register_action(
    name="check_pii",
    handler=lambda text: any([
        "身份证" in text,
        "手机号" in text,
        "银行卡" in text,
    ])
)

response = rails.generate(
    messages=[{"role": "user", "content": "我的身份证号是 110101199001011234"}],
)

print(response)
# Output: "抱歉，请勿分享个人敏感信息。"
```

```python
# Guardrails AI 用法
from guardrails import Guard
from guardrails.hub import ToxicLanguage, PIIFilter, RegexMatch

guard = Guard().use_many(
    ToxicLanguage(threshold=0.7, on_fail="fix"),
    PIIFilter(on_fail="filter"),
    RegexMatch(regex=r"^[^<]*$", on_fail="reask"),
)

response = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一封邮件"}],
    max_tokens=500,
)

print(response.validation_passed)
print(response.guarded_output)
```

### LLM API 成本对比

| 模型 | 输入 ($/M tokens) | 输出 ($/M tokens) | 上下文窗口 | 适用场景 |
|------|------------------|------------------|-----------|---------|
| GPT-4o | $2.50 | $10.00 | 128K | 复杂任务、代码 |
| GPT-4o-mini | $0.15 | $0.60 | 128K | 简单任务、分类 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 200K | 长文档分析 |
| DeepSeek-V3 | $0.27 | $1.10 | 128K | 性价比首选 |
| DeepSeek-R1 | $0.55 | $2.19 | 128K | 数学、推理 |
| Gemini 1.5 Pro | $1.25 | $5.00 | 1M | 超长上下文 |
| LLaMA 3.1 405B | $0.85 | $0.85 | 128K | 自托管性价比 |

### 案例：基于 LangFuse 的全链路追踪与成本归因

下面演示如何在调用 LLM 时用 LangFuse 装饰器自动记录 Prompt、Token 消耗、延迟与成本，实现按业务线的成本归因。

```python
# LangFuse 全链路追踪 + 成本归因
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai  # 自动上报的 OpenAI 封装

@observe(name="customer-service-pipe", as_type="generation")
def answer_question(context: str, question: str) -> str:
    prompt = f"上下文：{context}\n问题：{question}\n请给出准确简洁的回答。"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content

# 1. 调用并记录元数据（业务线、用户）
langfuse_context.update_current_trace(
    user_id="user_1001",
    metadata={"business_line": "ecommerce", "app_version": "2.3.0"},
)

result = answer_question("退货政策：7 天无理由退货。", "如何申请退货?")

# 2. 强制刷新上报
langfuse_context.flush()
```

### 实现案例：语义缓存降低 50% 调用成本

```python
# 基于向量相似度的语义缓存（GPTCache 风格）
import hashlib
import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.92):
        self.threshold = threshold
        self.keys = []          # 历史 query 向量
        self.values = []        # 历史回答

    def _embed(self, text: str):
        # 实际可替换为 sentence-transformers；此处用哈希占位演示
        h = hashlib.md5(text.encode("utf-8")).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(float)
        norm = np.linalg.norm(vec)
        return vec / norm

    def get(self, query: str) -> str | None:
        if not self.keys:
            return None
        q = self._embed(query)
        sims = [float(np.dot(q, k)) for k in self.keys]
        best = max(sims)
        if best >= self.threshold:
            idx = int(np.argmax(sims))
            return self.values[idx]
        return None

    def put(self, query: str, answer: str):
        self.keys.append(self._embed(query))
        self.values.append(answer)

cache = SemanticCache(threshold=0.92)
cached = cache.get("如何申请退货")
if cached is None:
    ans = answer_question("退货政策：7 天无理由退货。", "如何申请退货?")
    cache.put("如何申请退货", ans)
```

### 成本控制策略对比

| 策略 | 实现复杂度 | 典型节省 | 质量风险 | 适用场景 |
|------|-----------|---------|---------|---------|
| 语义缓存 | 中 (需向量库) | 30-60% | 低 | 高频相似 Query |
| 模型路由 | 中 | 40-70% | 低 | 任务难度分层 |
| Prompt 压缩 | 低 | 50-80% | 极低 | 长系统 Prompt |
| 批量异步 | 低 | 20-30% | 无 | 离线/非实时 |
| KV Cache 前缀复用 | 中 | 30-50% | 无 | 固定系统 Prompt |
| 输出长度裁剪 | 低 | 10-50% | 需权衡 | 可控输出格式 |

### Mermaid: 成本治理闭环

```mermaid
flowchart TD
    A["用户请求"] --> B["语义缓存命中?"]
    B -->|是| C["直接返回缓存"]
    B -->|否| D["模型路由判断难度"]
    D -->|简单| E["小模型 gpt-4o-mini"]
    D -->|复杂| F["大模型 gpt-4o"]
    E --> G["记录 Token / 成本"]
    F --> G
    G --> H["LangFuse 成本看板"]
    H --> I{"超预算?"}
    I -->|是| J["限流 / 降级"]
    I -->|否| A
    C --> A
```

## 3. RAG 观察

### 质量监控指标

| 指标 | 衡量内容 | 计算方法 | 目标值 |
|------|---------|---------|-------|
| 检索命中率 | 相关文档是否被召回 | Hit Rate @ K | > 95% |
| NDCG | 排序质量 | 归一化折损累计增益 | > 0.9 |
| MRR | 第一个相关结果位置 | 平均倒数排名 | > 0.85 |
| 忠实度 | 回答是否基于上下文 | LLM-as-Judge | > 90% |
| 回答相关性 | 回答是否针对问题 | 语义相似度 | > 0.8 |
| 端到端延迟 | 检索+生成总耗时 | P50/P99 | < 2s / < 5s |

## 4. 成本管理

### Token 计价模式

| 模型级别 | 输入价格 ($/M) | 输出价格 ($/M) | 典型月调用 |
|---------|---------------|---------------|-----------|
| 旗舰 (GPT-4o, Claude 3.5) | $2.50 - $3.00 | $10.00 - $15.00 | $50K+
| 性价比 (GPT-4o-mini, DeepSeek) | $0.15 - $0.55 | $0.28 - $2.19 | $5K+
| 自托管 (LLaMA-70B) | $0.85 (GPU折旧) | $0.85 | $10K+

### 成本优化

| 策略 | 节省比例 | 实现方式 | 对质量影响 |
|------|---------|---------|-----------|
| Prompt 压缩 | 50-80% | 删除冗余、合并指令 | 极小 (结构优化) |
| 缓存常用回复 | 30-60% | 语义缓存 (相似 query 命中) | 无 |
| 模型路由 | 40-70% | 简单任务用小模型 | 无 (任务适配) |
| 异步批量 | 20-30% | 离线任务批量处理 | 无 |
| KV Cache 复用 | 30-50% | 系统 prompt 前缀缓存 | 无 |
| 输出长度限制 | 10-50% | max_tokens 控制 | 需权衡完整性 |

### 成本监控

```bash
# 使用 Helicone / LangFuse 追踪
curl -X POST https://api.helicone.ai/v1/log \
    -H "Authorization: Bearer $HELICONE_KEY" \
    -d '{"model": "gpt-4o", "prompt_tokens": 500, "completion_tokens": 200}'

# LangFuse 追踪
langfuse.trace(
    name="chat-completion",
    usage={"input": 500, "output": 200, "unit": "TOKENS"},
    cost=0.00375,
)
```

## 5. 护栏 Guardrails

### 护栏类型

| 类型 | 方向 | 检测内容 | 处理方式 |
|------|------|---------|---------|
| 输入护栏 | 输入 | 提示注入、越狱、敏感词 | 拦截 / 替换 |
| 输出护栏 | 输出 | 有害内容、偏见、幻觉 | 过滤 / 重生成 |
| 上下文护栏 | 上下文 | PII、商业秘密、合规 | 脱敏 / 拦截 |
| 格式护栏 | 输出 | JSON Schema、长度 | 校验 / 重试 |

### 护栏实现对比

| 工具 | 开源 | 输入/输出 | 自定义规则 | LLM 评估 | 集成方式 |
|------|------|----------|-----------|---------|---------|
| NeMo Guardrails | ✅ | 双向 | ✅ Colang 语言 | ✅ | Python SDK |
| Guardrails AI | ✅ | 双向 | ✅ XML/JSON 规范 | ✅ | Python SDK |
| NVIDIA 内容安全 | ✅ | 输出 | ❌ 预训练 | ✅ | API |
| Azure AI Content Safety | ❌ | 输出 | ✅ 自定义 | ✅ | REST API |
| OpenAI Moderation | ❌ | 双向 | ❌ 固定类别 | ✅ | API |

## 6. 用户反馈 Pipeline

```mermaid
flowchart TD
    subgraph 用户交互
    A[用户输入] --> B[LLM 调用]
    B --> C[生成回复]
    C --> D[展示给用户]
    D --> E[用户反馈]
    end
    subgraph 数据收集
    E -->|点赞/点踩| F[反馈日志]
    E -->|举报| G[人工审核队列]
    B --> H[全量日志]
    H --> I[(数据湖)]
    end
    subgraph 分析层
    F --> J[质量评分计算]
    J --> K{评分低于阈值?}
    K -->|是| L[标记为需改进]
    G --> M[审核结果]
    I --> N[离线分析]
    N --> O[问题模式识别]
    end
    subgraph 改进层
    L --> P[Prompt 优化]
    M --> P
    O --> Q[检索案例库更新]
    O --> R[Fine-tuning 数据]
    P --> S[新 Prompt 版本 A/B]
    S --> T[对比线上指标]
    T -->|提升| U[全量上线]
    T -->|下降| P
    end
```

### 监控全链路

| 阶段 | 监控指标 | 采集方式 | 告警阈值 |
|------|---------|---------|---------|
| 输入 | Token 数、延迟、请求率 | API Gateway | P99 > 1s |
| 检索 | 检索延迟、召回数量、分数 | RAG 中间件 | 检索 > 200ms |
| 生成 | TTFT (首 token 延迟)、Token 速率 | 推理引擎 | TTFT > 2s |
| 输出 | 输出 Token 数、内容安全 | Guardrails | 安全触发 > 1% |
| 反馈 | 点赞率、举报率、用户留存 | 用户行为分析 | 点赞率 < 50% |

### Shell: LLMOps 部署

```bash
# LangFuse 自托管
docker run -d --name langfuse \
    -p 3000:3000 \
    -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/langfuse \
    -e NEXTAUTH_SECRET=my-secret \
    -e NEXTAUTH_URL=http://localhost:3000 \
    langfuse/langfuse:latest

# Guardrails 配置示例
cat > config/guardrails/config.yml << 'EOF'
models:
  - type: main
    engine: openai
    model: gpt-4o
rails:
  input:
    flows:
      - check jailbreak
      - check sensitive topics
  output:
    flows:
      - check toxicity
      - check factual consistency
      - check pii
EOF

# Helicone 代理 vLLM
docker run -d --name helicone-proxy \
    -p 8080:8080 \
    -e HELICONE_API_KEY=$KEY \
    -e OPENAI_API_BASE=http://localhost:8000 \
    helicone/helicone:latest
```

## 7. 2025-2026 趋势
- **LLMOps 平台化**：一站式 Prompt/评估/监控/成本管理
- **自动评估器**：LLM-as-Judge 评估，减少人工标注
- **在线学习**：用户反馈实时微调，DPO/ORPO 在线更新
- **多模态 Ops**：文本+图像+语音统一运维
- **Agent Ops**：AI Agent 行为监控、工具调用审计
- **成本治理平台**：Token 预算、模型路由、自动降级
