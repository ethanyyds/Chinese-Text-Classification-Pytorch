# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Chinese text classification using PyTorch. 7 models (TextCNN, TextRNN, TextRNN_Att, TextRCNN, FastText, DPCNN, Transformer) trained on THUCNews dataset (200k Chinese news titles, 10 categories).

### Running Models

```bash
python3 run.py --model TextCNN          # or TextRNN, TextRNN_Att, TextRCNN, DPCNN, Transformer
python3 run.py --model FastText --embedding random   # FastText uses random embeddings
```

See `README.md` for full usage and expected accuracy per model.

### Key Notes

- **No GPU required.** Training auto-falls back to CPU. CPU training for TextCNN takes ~12 minutes on the Cloud VM.
- **No `requirements.txt` in repo.** Dependencies: `torch` (CPU), `tqdm`, `scikit-learn`, `tensorboardX`, `numpy`. Install via: `pip install torch --index-url https://download.pytorch.org/whl/cpu tqdm scikit-learn tensorboardX numpy`
- **No linter or test suite exists** in this repo. Validation is done by running training and checking accuracy against README benchmarks.
- **Use `python3 -u`** (unbuffered) when piping output to `tee` or logs, otherwise training progress will not appear until the buffer flushes.
- **Early stopping**: training auto-stops if validation loss doesn't improve for 1000 batches (`config.require_improvement`).
- **TensorBoard** logs are written to `THUCNews/log/`. Optionally run `tensorboard --logdir THUCNews/log/` to visualize.
- **Model checkpoints** are saved to `THUCNews/saved_dict/`.
