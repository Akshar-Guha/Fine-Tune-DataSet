# 🚀 ModelOps: The Laptop-First LLM Fine-Tuning Platform

![ModelOps Banner](https://img.shields.io/badge/ModelOps-Laptop%20Edition-purple?style=for-the-badge) ![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

**ModelOps** is a unified, lightweight, and local-first platform designed to democratize LLM fine-tuning. It brings enterprise-grade orchestration to your personal machine, allowing you to fine-tune, manage, and serve 1B-7B parameter models without needing a massive cloud cluster.

---

## 🛑 The Real-World Problem

Training and managing Large Language Models (LLMs) has historically been the privilege of large enterprises with massive GPU clusters. For independent developers, researchers, and small startups, the barrier to entry is high:

*   ❌ **Complex Toolchains**: Stitching together Hugging Face, PyTorch, LoRA, quantization scripts, and vector DBs is fragile and time-consuming.
*   ❌ **Resource Constraints**: Most standard MLOps stacks (Kubeflow, etc.) are too heavy for a single workstation.
*   ❌ **Data Chaos**: Keeping track of dataset versions, prompt templates, and cleaning rules locally often ends in a mess of "final_final_v2.csv" files.
*   ❌ **Black Box Training**: Running scripts in a terminal gives you little insight into loss curves or potential overfitting until it's too late.

## ⚔️ Why ModelOps?

You might check out other tools, but here is why **ModelOps** is the only **End-to-End Studio** for local developers:

| Feature | **ModelOps** | LM Studio / Jan.ai | Axolotl / LLaMA Factory | Weights & Biases |
| :--- | :---: | :---: | :---: | :---: |
| **Local Inference UI** | ✅ | ✅ | ❌ | ❌ |
| **GUI-Based Training** | ✅ | ❌ | ❌ | ❌ |
| **Data Cleaning & Management** | ✅ | ❌ | ❌ | ❌ |
| **Job Orchestration** | ✅ | ❌ | ⚠ (CLI only) | ❌ |
| **Purpose** | **Complete Lifecycle Studio** | Chat / Inference | Hardcore Training | Monitoring |

We are not just a wrapper around a script; we are a platform that takes you from **Raw Data → Cleaned Dataset → Fine-Tuned Model → Production Deployment**.

## 🖥️ Hardware Reality Check

Honesty is key. Fine-tuning requires resources, but we've optimized ModelOps to squeeze every bit of performance out of consumer hardware.

| Tier | Hardware Example | Capabilities |
| :--- | :--- | :--- |
| **Minimum** | **NVIDIA RTX 3060 (8GB VRAM)** | Fine-tune **1B-3B models** (TinyLlama, StableLM) with QLoRA. |
| **Recommended** | **NVIDIA RTX 3090 / 4090 (24GB VRAM)** | Fine-tune **7B-13B models** (Llama 3, Mistral) comfortably. |
| **CPU Only** | Apple M-Series / High-end Intel | **Inference & Data Processing Only**. Training is not viable on CPU alone. |

## 🔮 The Vision & Long-Term Moat

We are building more than just a tool; we are building an ecosystem for the "Local AI" era:

1.  **Community Registry**: Future updates will let you share your cleaned datasets and training recipes with the community.
2.  **One-Click GGUF Export**: Seamlessly convert your adapters to `.gguf` and push directly to Ollama.
3.  **Active Learning**: A feedback loop where your manual chat corrections automatically become new training data.

---

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI, SQLAlchemy, SQLite
*   **Frontend**: React, TypeScript, TailwindCSS, Vite
*   **ML Core**: PyTorch, Hugging Face Transformers, PEFT, BitsAndBytes
*   **Orchestration**: Background task management for reliable long-running jobs.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 16+
*   NVIDIA GPU (Recommended for training) or CPU (for inference/testing)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Akshar-Guha/Fine-Tune-DataSet.git
    cd Fine-Tune-DataSet/modelops
    ```

2.  **Run the super-starter**
    We've included a script to set up everything (Python venv, Node dependencies) and launch both servers.
    ```bash
    ./run_all.bat
    ```

3.  **Access the Dashboard**
    Open [http://localhost:5173](http://localhost:5173) in your browser.

---

*Built with ❤️ to make AI accessible.*
