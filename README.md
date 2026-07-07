# AI 知识库

AI 技术栈系统性知识分类文档集。覆盖从机器学习基础到 AGI 前沿的完整知识体系，适合入门、进阶和专家各层次学习者。

## 完整目录

```
ai技术栈/
│
├── 00-机器学习基础/          # ML 算法基石
│   ├── README.md             # 总览与学习路线
│   ├── 01-监督学习.md        # 回归/分类/SVM/决策树/集成/XGBoost
│   ├── 02-无监督学习.md      # 聚类/K-Means/DBSCAN/GMM/降维/PCA/t-SNE/UMAP
│   ├── 03-模型评估与调优.md  # 指标/K折/超参搜索/偏差方差/Bayes优化
│
├── 01-深度学习/              # DL 理论基础
│   ├── README.md             # 总览与架构图谱
│   ├── 01-神经网络基础.md    # 感知机/BP/激活函数/初始化/正则化
│
├── 02-自然语言处理NLP/       # NLP 传统任务
│   ├── README.md             # 任务分类与发展脉络
│   ├── 01-文本表示与嵌入.md  # Word2Vec/BERT/SBERT/BGE/MTEB
│
├── 03-大语言模型LLM/         # ★ 大模型全栈（已完整）
│   ├── 01-模型架构演进.md    # Transformer→GPT→MoE→CSA+HCA→mHC
│   ├── 02-预训练技术.md      # 数据/Muon优化器/合成数据Scaling/GRPO
│   ├── 03-微调与对齐.md      # LoRA/DoRA/QLoRA/GRPO/DPO/多阶段蒸馏
│   ├── 04-推理优化.md        # KV Cache/CSA+HCA/FlashAttention/SSD/EAGLE3
│   ├── 05-提示工程.md        # CoT/ToT/ReAct/攻防/结构化输出
│   ├── 06-检索增强生成RAG.md # GraphRAG/Agentic RAG/多模态/Contextual
│   ├── 07-AI智能体Agent.md   # MCP 2026规范/A2A/ADK/Function Calling
│   ├── 08-评估与基准.md      # MMLU-Pro/GPQA/SWE-bench Pro/Arena
│   └── 09-推理框架与部署.md  # vLLM/SGLang/TensorRT-LLM/Ollama
│
├── 04-计算机视觉CV/          # CV 完整知识
│   ├── README.md             # 总览与框架对比
│   ├── 02-图像分类.md        # AlexNet→ResNet→ViT→ConvNeXt
│   ├── 03-目标检测.md        # R-CNN/YOLO/DETR/DINO/评估指标
│   ├── 05-图像生成.md        # GAN→扩散模型→ControlNet→视频生成
│
├── 05-语音与音频/            # 语音技术
│   ├── README.md             # 总览与指标
│   ├── 01-语音识别ASR.md     # CTC/RNN-T/Whisper/SenseVoice/评估
│   ├── 02-语音合成TTS.md     # Tacotron/FastSpeech/VITS/CosyVoice
│
├── 06-强化学习与决策/        # RL 从基础到前沿
│   ├── README.md             # 核心概念
│   ├── 01-基础强化学习.md    # MDP/Q-Learning/SARSA/TD
│
├── 07-AI工程与基础设施/      # AI 工程化
│   ├── README.md             # MLOps与成熟度模型
│   ├── 01-MLOps与实验管理.md # 实验/流水线/监控/特征存储/LLMOps
│
├── 08-AI安全与伦理/          # 负责任的 AI
│   ├── README.md             # 风险分类
│   ├── 01-模型安全.md        # 对抗攻击/越狱/提示注入/防御
│
├── 09-行业应用/              # 行业落地
│   ├── README.md             # 应用模式矩阵
│   ├── 03-AI编程.md          # 代码模型/AI IDE/Agent编程
│
├── 10-前沿方向/              # 技术前沿
│   ├── README.md             # 2026趋势栈
│   └── 07-2026前沿趋势.md    # 年度全景/展望2027
│
└── README.md                 # 当前文件
```

## 阅读建议

| 角色 | 建议路径 |
|------|---------|
| 🟢 入门 | 00-基础 → 01-深度学习 → 03-LLM 基础概念 |
| 🔵 进阶 | 03-LLM 全模块 → 04-CV → 05-语音 → 06-RL |
| 🔴 大师 | 07-工程 → 08-安全 → 10-前沿 → 论文精读 |

## 覆盖模型（截至 2026.7）

| 厂商 | 模型 |
|------|------|
| DeepSeek | V3/R1/V3.2/V4-Pro/V4-Flash |
| Meta | Llama 3/4 Scout/4 Maverick/4 Behemoth |
| Alibaba | Qwen 2.5/3/3.5/3.7 Max |
| Google | Gemini 2.5/3/3.1 Pro |
| Anthropic | Claude 3.5/4/Opus 4.5-4.8/Fable 5 |
| 其他 | Mistral Large 3/Kimi K2/GLM-5/Nemotron 3/Phi-4 |

## 知识统计

- **总文档数**：25+
- **覆盖知识点**：1000+
- **最新更新**：2026年7月
- **许可证**：知识共享

---

*"AI 不只是模型，是系统工程。"*
