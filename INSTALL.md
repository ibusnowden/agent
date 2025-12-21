# COEVOLVE Installation Guide

Complete setup instructions for the Co-Evolutionary Multi-Agent Learning Framework.

---

## Quick Start

```bash
cd ~/Desktop/COEVOLVE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

---

## Prerequisites

### System Requirements

- **Python:** 3.9 or higher (3.10+ recommended)
- **RAM:** 16GB minimum (32GB recommended for local models)
- **GPU:** Optional but recommended for local model training
- **Storage:** 50GB free space (for models and data)

### Required Accounts (for API access)

1. **OpenAI** (recommended for quick start)
   - Sign up: https://platform.openai.com/signup
   - Get API key: https://platform.openai.com/api-keys
   - Set billing: https://platform.openai.com/account/billing

2. **Anthropic** (optional, for Claude models)
   - Sign up: https://console.anthropic.com/
   - Get API key from dashboard

3. **Weights & Biases** (for experiment tracking)
   - Sign up: https://wandb.ai/signup
   - Get API key: https://wandb.ai/authorize

4. **HuggingFace** (for local models)
   - Sign up: https://huggingface.co/join
   - Get token: https://huggingface.co/settings/tokens

---

## Detailed Installation

### Step 1: Clone and Setup Environment

```bash
cd ~/Desktop/COEVOLVE

# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Core Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import langchain, chromadb, torch; print('Core dependencies OK!')"
```

### Step 3: Configure API Keys

Create `.env` file:

```bash
cat > .env << EOF
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic (optional)
ANTHROPIC_API_KEY=your-key-here

# Weights & Biases (optional)
WANDB_API_KEY=your-key-here

# HuggingFace (optional)
HF_TOKEN=your-token-here

# Model Settings
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4

# Paths
DATA_DIR=./data
EXPERIMENTS_DIR=./experiments
LOGS_DIR=./logs
EOF
```

### Step 4: Download Embedding Models

```bash
# Download sentence transformer for embeddings
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model downloaded!')
"
```

### Step 5: Initialize Database

```bash
# Create data directories
mkdir -p data/memory data/debates data/code_patches data/preferences

# Initialize ChromaDB
python << EOF
import chromadb
client = chromadb.PersistentClient(path="data/memory")
print("ChromaDB initialized!")
EOF
```

### Step 6: Verify Installation

```bash
# Run test script
python tests/test_installation.py
```

---

## Optional: Local Model Setup

### Option 1: VLLM (Fast Inference)

```bash
# Install VLLM
pip install vllm

# Download a model (example: Llama-2-7B)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# Start VLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

Update `.env`:
```bash
LOCAL_MODEL_URL=http://localhost:8000/v1
```

### Option 2: Llama.cpp (CPU Inference)

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Download GGUF model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Move to models directory
mkdir -p models
mv llama-2-7b-chat.Q4_K_M.gguf models/
```

---

## Framework-Specific Setup

### LangGraph (State Machine)

```bash
# Verify LangGraph installation
python -c "from langgraph.graph import StateGraph; print('LangGraph OK!')"
```

### AutoGen (Multi-Agent)

```bash
# Test AutoGen
python << EOF
from autogen import AssistantAgent, UserProxyAgent
print("AutoGen installed successfully!")
EOF
```

### Weights & Biases

```bash
# Login to W&B
wandb login

# Test logging
python << EOF
import wandb
wandb.init(project="coevolve-test", name="installation-test")
wandb.log({"test": 1})
wandb.finish()
print("W&B configured!")
EOF
```

### MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000 &

# Open browser to http://localhost:5000
echo "MLflow UI: http://localhost:5000"
```

---

## GPU Setup (Optional)

### For NVIDIA GPUs

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Mac (Apple Silicon)

```bash
# PyTorch with MPS (Metal Performance Shaders)
pip install torch torchvision torchaudio

# Verify MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Troubleshooting

### Issue: ImportError for LangGraph

```bash
pip install --upgrade langchain langgraph
```

### Issue: ChromaDB Persistence Error

```bash
# Clear and reinitialize
rm -rf data/memory
mkdir -p data/memory
python -c "import chromadb; chromadb.PersistentClient(path='data/memory')"
```

### Issue: OpenAI API Key Not Found

```bash
# Check .env file exists
cat .env | grep OPENAI_API_KEY

# Load in Python
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv('OPENAI_API_KEY', 'NOT FOUND'))
"
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Use smaller models (gpt-3.5-turbo instead of gpt-4)
2. Reduce batch size in config
3. Enable gradient checkpointing for training
4. Use quantized models (4-bit, 8-bit)

---

## Testing the Installation

Create `tests/test_installation.py`:

```python
#!/usr/bin/env python3
"""Test COEVOLVE installation."""

def test_imports():
    """Test all major imports."""
    print("Testing imports...")

    import langchain
    import langgraph
    import chromadb
    import torch
    import transformers
    import openai
    import anthropic
    import wandb

    print("✓ All imports successful!")

def test_models():
    """Test model loading."""
    print("\nTesting models...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode("test")
    assert embedding.shape[0] == 384

    print("✓ Embedding model works!")

def test_database():
    """Test ChromaDB."""
    print("\nTesting database...")

    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("test")
    collection.add(documents=["test"], ids=["1"])
    results = collection.query(query_texts=["test"], n_results=1)
    assert len(results['ids'][0]) == 1

    print("✓ ChromaDB works!")

def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")

    from core.config import CoEvolveConfig
    config = CoEvolveConfig()
    assert config.model.temperature == 0.7

    print("✓ Configuration system works!")

if __name__ == "__main__":
    test_imports()
    test_models()
    test_database()
    test_config()

    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("="*50)
    print("\nCOEVOLVE is ready to use!")
```

Run tests:
```bash
python tests/test_installation.py
```

---

## Next Steps

After successful installation:

1. **Read the Documentation**
   - Main README: `README.md`
   - Papers: `papers/01-05_*.md`
   - Progress: `docs/PROGRESS.md`

2. **Run Your First Experiment**
   ```bash
   python experiments/run_debate.py --config configs/quick_test.yaml
   ```

3. **Explore Examples**
   - Check `examples/` directory for tutorials
   - Start with simple debate scenario

4. **Join the Community**
   - Report issues on GitHub
   - Contribute novel game scenarios
   - Share experimental results

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove environment
rm -rf venv

# Remove data (optional)
rm -rf data experiments logs
```

---

## Support

- **Documentation:** See `docs/` directory
- **Issues:** GitHub Issues (when open-sourced)
- **Email:** [your-email@example.com]

---

**Installation complete! You're ready to explore co-evolutionary AI.**
