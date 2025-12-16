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

## ✅ The Solution: ModelOps

We built **ModelOps** to solve this by providing a cohesive "Studio" experience for the local developer. It abstracts away the complexity of the training loop, data management, and model versioning into a single, beautiful interface.

### ✨ Key Features

#### 1. 🧠 Local QLoRA Fine-Tuning
Fine-tune capable models (like TinyLlama, Phi-2, or Qwen) directly on your laptop using **QLoRA (Quantized Low-Rank Adaptation)**.
*   **4-bit Quantization**: Drastically reduces VRAM usage.
*   **Optimized Trainer**: Pre-configured defaults for varying hardware tiers.
*   **Real-time Monitoring**: Watch your loss curves and metrics live.

#### 2. 📂 Smart Registries
Stop losing track of your assets.
*   **Dataset Registry**: Search and download directly from Hugging Face, or upload local files. Includes automated **quality checks** and **auto-labeling** rules.
*   **Model Registry**: Manage your base models and fine-tuned adapters. Track metadata, chat templates, and quantization formats (GGUF, AWQ).

#### 3. ⚙️ Automated Data Pipelines
Good models start with good data.
*   **Data Cleaning**: Auto-strip whitespace, drop duplicates, and normalize labels.
*   **Quality Gates**: Automatically flag datasets that don't meet quality thresholds (e.g., too many missing values).

#### 4. 🤖 Instant Inference & Serving
Don't just train—test.
*   **Built-in Chat UI**: Chat with your fine-tuned model immediately after training.
*   **Ollama Integration**: Seamlessly offload inference to Ollama for optimized performance.
*   **Review Mode**: Compare model outputs side-by-side.

#### 5. 🕜 Job Orchestration
*   **Background Jobs**: Long-running training jobs are handled asynchronously. Close your browser and come back later.
*   **History**: A complete audit log of every training run, including hyperparameters and final metrics.

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
