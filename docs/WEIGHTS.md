# Model Weights

This document explains how model weights are distributed and options for managing them.

## Included Weights

| Model | File | Size | Storage |
|-------|------|------|---------|
| Neural 6502 | `weights/neural_cpu_best.pt` | 9.3 MB | Direct (Git) |
| BBDOS LM | Not included | 146 MB | Contact author |

## Why Direct Inclusion?

The Neural 6502 weights (9.3 MB) are stored directly in the Git repository because:

1. **Simplicity** - `git clone` gives you everything needed to run the demo
2. **Under GitHub's limit** - 9.3 MB is well below the 100 MB file limit
3. **Reproducibility** - No external dependencies or broken links

## Large Model Weights (BBDOS LM)

The language model weights (146 MB) exceed GitHub's file size limit. Options:

**Option A: Contact Author**
```
Email: iam@anjaustin.com
Subject: BBDOS LM Weights Request
```

**Option B: Git LFS** (if hosting your own copy)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "weights/*.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for weights"
```

## For Forks with Larger Models

If you train larger models and want to include weights in your fork:

1. **Under 100 MB** - Include directly (simplest)
2. **100 MB - 2 GB** - Use Git LFS
3. **Over 2 GB** - Use external hosting (HuggingFace Hub, S3, etc.)

### Git LFS Setup

```bash
# One-time setup
git lfs install

# Track PyTorch checkpoints
git lfs track "*.pt"
git lfs track "*.pth"

# Commit tracking config
git add .gitattributes
git commit -m "Track large files with LFS"

# Now add your weights normally
git add weights/my_large_model.pt
git commit -m "Add model weights"
git push
```

### HuggingFace Hub Alternative

For very large models or public distribution:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="weights/my_model.pt",
    path_in_repo="my_model.pt",
    repo_id="username/bbdos-weights",
    repo_type="model"
)
```

## Verifying Weight Integrity

```bash
# Check file sizes
ls -lh weights/

# Verify checkpoint loads
python -c "import torch; m = torch.load('weights/neural_cpu_best.pt'); print('OK')"
```

---

*See [README.md](../README.md) for usage instructions.*
