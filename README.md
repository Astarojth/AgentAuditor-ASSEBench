<p align="center">
  <img src="assets/logo.png" alt="AgentAuditor Logo" width="150"/>
</p>

<h3 align="center">🕵️ AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2506.00641">📜 Paper</a> |
  <a href="https://github.com/Astarojth/AgentAuditor-ASSEBench/tree/main/AgentAuditor">📚 Dataset</a> |
  <a href="#-quick-start">🚀 Quick Start</a>
</p>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=Astarojth.AgentAuditor-ASSEBench" alt="Visitor Badge" />
  <img src="https://img.shields.io/github/stars/Astarojth/AgentAuditor-ASSEBench?style=social" alt="GitHub Stars" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version" />
</p>

---

> **AgentAuditor** is a universal, training-free, memory-augmented reasoning framework that empowers LLM evaluators to emulate human expert evaluators in identifying safety and security risks in LLM agent interactions.

## 💥 News

- **[2025-06-03]** We release the AgentAuditor paper along with dataset! 🚀


## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Repository Structure](#-repository-structure)
- [🧪 Pipeline Overview](#-pipeline-overview)
- [⚡ Quick Start](#-quick-start)
- [📄 Citation](#-citation)
- [📬 Contact](#-contact)


## 🔍 Overview

As Large Language Model (LLM)-based agents become increasingly autonomous, they introduce new safety and security risks that traditional evaluation methods struggle to detect. **AgentAuditor** addresses this critical challenge through a novel approach that combines experiential memory with human-like reasoning.

### The Problem
- **Autonomous LLM agents** are deployed in high-stakes scenarios (finance, healthcare, critical infrastructure)
- **Traditional evaluation methods** fail to capture nuanced safety and security risks
- **Human-level assessment** is needed but expensive and doesn't scale

### Our Solution
AgentAuditor introduces a sophisticated evaluation framework that:

1. **🧠 Builds Experiential Memory**: Constructs a structured knowledge base from past agent interactions, extracting semantic features (scenario, risk, behavior) and generating Chain-of-Thought (CoT) reasoning traces

2. **🔍 Employs Smart Retrieval**: Uses multi-stage, context-aware retrieval-augmented generation (RAG) to guide LLM evaluators with relevant historical experiences

3. **📊 Introduces ASSEBench**: The first comprehensive benchmark specifically designed to evaluate how well LLM-based evaluators can identify both safety risks and security threats in agent interactions

## ✨ Key Features

- **🚀 Training-Free**: No model fine-tuning required - works with any LLM evaluator
- **🎯 Human-Level Performance**: Achieves expert-level accuracy in safety assessment  
- **📈 Scalable**: Automated pipeline that scales to large datasets
- **🔧 Modular Design**: Each component can be used independently or as part of the full pipeline
- **🌐 Universal**: Works across different agent types and interaction scenarios
- **📊 Comprehensive**: Evaluates both safety risks and security threats

## 🏗️ Architecture

![Overall Architecture](assets/agent_auditor_overview.png)
![Overall Architecture](assets/workflow.png)

## 🏗️ ASSEBench Category

![Overall Architecture](assets/dataset.png)

## 📁 Repository Structure

```
AgentAuditor-ASSEBench/
├── 📄 README.md                        # Project documentation
├── 📄 LICENSE                          # Apache 2.0 license
├── 📄 requirements.txt                  # Python dependencies
├── 🚀 agent_auditor.sh                 # Main AgentAuditor pipeline script
├── 🎯 direct_eval.sh                   # Direct evaluation baseline script
├── 🎨 assets/                          # Visual assets
│   ├── logo.png                        # Project logo
│   └── agent_auditor_overview.png      # Architecture diagram
├── 🔧 AgentAuditor/                    # Core framework implementation
│   ├── __init__.py                     # Package initialization
│   ├── __main__.py                     # CLI entry point (python -m AgentAuditor)
│   ├── 📊 data/                        # Training and configuration data
│   │   ├── agentharm.json              # AgentHarm dataset
│   │   ├── AgentJudge-*.json           # AgentJudge dataset variants
│   │   ├── rjudge.json                 # RJudge dataset
│   │   └── fewshot.txt                 # Few-shot CoT examples
│   ├── ⚙️ params/                      # Pre-computed parameters
│   │   ├── clus_param.pkl              # Clustering parameters (FINCH)
│   │   └── infer_param.pkl             # Inference parameters (embeddings)
│   └── 🛠️ tasks/                       # Pipeline task implementations
│       ├── preprocess.py               # LLM-based semantic annotation
│       ├── cluster.py                  # Weighted clustering with FINCH
│       ├── demo.py                     # CoT demonstration generation
│       ├── demo_repair.py              # CoT validation and repair
│       ├── infer_emb.py                # Embedding-based retrieval
│       ├── infer.py                    # Few-shot inference engine
│       ├── infer_fix1.py               # JSON parsing and correction
│       ├── infer_fix2.py               # LLM-based output refinement
│       ├── eval.py                     # Performance evaluation
│       ├── direct_eval.py              # Zero-shot baseline evaluation
│       └── direct_metric.py            # Baseline metrics calculation
└── 📊 ASSEBench/                       # Benchmark dataset
    ├── category/                       # Risk categorization
    │   ├── safety/                     # Safety-related scenarios
    │   │   ├── f.json                  # Failure cases
    │   │   ├── r.json                  # Risk cases  
    │   │   └── s.json                  # Success cases
    │   └── security/                   # Security-related scenarios
    │       ├── f.json                  # Failure cases
    │       ├── r.json                  # Risk cases
    │       └── s.json                  # Success cases
    └── dataset/                        # Evaluation datasets
        ├── AgentJudge-loose.json       # Loose evaluation criteria
        ├── AgentJudge-safety.json      # Safety-focused evaluation
        ├── AgentJudge-security.json    # Security-focused evaluation
        └── AgentJudge-strict.json      # Strict evaluation criteria
```


## 🧪 Pipeline Overview

AgentAuditor employs a sophisticated multi-stage pipeline that mimics human expert evaluation. The framework transforms raw agent interactions into structured assessments through memory-augmented reasoning.

### 🔄 Main AgentAuditor Pipeline

The complete pipeline consists of six sequential stages:

| Stage | Component | Purpose | Input | Output |
|-------|-----------|---------|--------|---------|
| **1️⃣ Preprocessing** | `preprocess.py` | Semantic annotation using LLM | Raw agent interactions | Structured memory with scenario/risk/behavior tags |
| **2️⃣ Clustering** | `cluster.py` | Identify representative cases | Annotated interactions | Cluster representatives for demonstrations |
| **3️⃣ Demo Generation** | `demo.py` + `demo_repair.py` | Create & validate CoT examples | Representative cases | High-quality CoT demonstrations |
| **4️⃣ Retrieval** | `infer_emb.py` | Find relevant experiences | Test cases + demo pool | Test cases + similar examples |
| **5️⃣ Inference** | `infer.py` + `infer_fix*.py` | Few-shot evaluation with CoT | Augmented test cases | Safety predictions with reasoning |
| **6️⃣ Evaluation** | `eval.py` | Performance analysis | Predictions vs ground truth | Metrics (Accuracy, F1, etc.) |

### 🎯 Direct Evaluation Baseline

For comparison without AgentAuditor enhancement:

| Component | Purpose | Method |
|-----------|---------|---------|
| `direct_eval.py` | Zero-shot safety evaluation | Direct LLM assessment |
| `direct_metric.py` | Baseline performance metrics | Standard evaluation metrics |

### 💡 Usage Patterns

**🚀 Complete Pipeline:**
```bash
bash agent_auditor.sh
```

**🔧 Individual Stages:**
```bash
# Semantic annotation
python -m AgentAuditor rjudge preprocess  

# Find representative cases
python -m AgentAuditor rjudge cluster     

# Generate demonstrations
python -m AgentAuditor rjudge demo        

# Embedding-based retrieval
python -m AgentAuditor rjudge infer_emb   

# Few-shot inference
python -m AgentAuditor rjudge infer       

# Calculate final metrics
python -m AgentAuditor rjudge eval        
```

**📊 Baseline Comparison:**
```bash
bash direct_eval.sh
```

## ⚡ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Astarojth/AgentAuditor-ASSEBench.git
cd AgentAuditor-ASSEBench

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

You need to configure your OpenAI API key in multiple locations. Look for `GPTConfig` classes throughout the codebase and update the `API_KEY` field:

```python
# Example configuration in Python files
class GPTConfig:
    API_KEY = "your-openai-api-key-here"
    MODEL = "gpt-4"  # or your preferred model
    # ... other settings
```

### 3. Run Evaluation

Choose your evaluation approach:

**🎯 Quick Evaluation (Full Pipeline):**
```bash
# Run complete AgentAuditor evaluation
bash agent_auditor.sh

# Run baseline comparison
bash direct_eval.sh
```

**🔧 Custom Dataset Evaluation:**
```bash
# Replace 'rjudge' with your dataset name
python -m AgentAuditor your_dataset preprocess
python -m AgentAuditor your_dataset cluster
python -m AgentAuditor your_dataset demo
python -m AgentAuditor your_dataset infer_emb
python -m AgentAuditor your_dataset infer
python -m AgentAuditor your_dataset eval
```

### 4. View Results

Results will be saved in the respective output directories. Key metrics include:
- **Accuracy**: Overall correctness of safety assessments
- **Precision/Recall**: Fine-grained performance analysis
- **F1-Score**: Balanced performance measure

## 📄 Citation

If you use AgentAuditor or ASSEBench in your research, please cite our paper:

```bibtex
@article{luo2025agentauditor,
  title={AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents}, 
  author={Hanjun Luo and Shenyu Dai and Chiming Ni and Xinfeng Li and Guibin Zhang and Kun Wang and Tongliang Liu and Hanan Salam},
  journal={arXiv preprint arXiv:2506.00641},
  year={2025},
  url={https://arxiv.org/abs/2506.00641}
}
```

## 📬 Contact

For questions, collaborations, or support:

**Primary Contacts:**
- **Hanjun Luo**: [hanjunluowork@gmail.com](mailto:hanjunluowork@gmail.com)
- **Chiming Ni**: [chimingni@gmail.com](mailto:chimingni@gmail.com)

**Project Resources:**
- 📄 **Paper**: [ArXiv Link](https://arxiv.org/abs/2506.00641)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Astarojth/AgentAuditor-ASSEBench/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Astarojth/AgentAuditor-ASSEBench/discussions)
- 🌐 **Project Page**: *Coming Soon*

---

<p align="center">
  <strong>Made with ❤️ for safer AI agents</strong><br/>
  <sub>AgentAuditor © 2025 - Licensed under Apache 2.0</sub>
</p>
