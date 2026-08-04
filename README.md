# Machine Unlearning: SISA vs. Fine-Tuning

This project implements and compares two machine unlearning strategies — **SISA (Sharded, Isolated, Sliced, and Aggregated) training** and **fine-tuning-based unlearning** — by measuring how effectively and efficiently each method can make a trained neural network "forget" a specific class, without retraining entirely from scratch.

Two experiments are run:

1. **MNIST + SISA unlearning** — forgetting digit class `7`.
2. **CIFAR-10 + fine-tuning unlearning** — forgetting the `airplane` class.

For each experiment, three models are trained and compared:

- **Original** — trained on the full dataset (including the class to be forgotten).
- **Unlearned** — the original model after applying the unlearning procedure (SISA shard-retraining or fine-tuning on retained data).
- **Retrained** — a model trained from scratch using only the retained data (the "gold standard" baseline for what true forgetting should look like).

## Motivation

Machine unlearning addresses situations where specific training data must be removed from a model's influence (e.g., privacy requests, "right to be forgotten" regulations, or correcting for a class that must be removed) without the cost of retraining the entire model from zero. This project empirically evaluates the trade-off between **unlearning speed** and **resulting accuracy/forgetting quality**.

## Methods

### 1. SISA Unlearning (MNIST)

SISA splits the training set into shards, each handled by an independently trained model whose predictions are aggregated at inference time.

- The MNIST training set is split into **10 shards**.
- Samples from the forget class (`7`) are distributed across the first **5 shards**.
- **Original model**: each shard is trained independently (shards containing the forget class trained for 10 epochs, the rest for 3 epochs), then predictions are aggregated across all shard models.
- **Unlearning**: only the shards that contained forget-class samples are retrained after removing those samples — the other shards are left untouched, which is the main efficiency advantage of SISA.
- **Retrained baseline**: all shards are trained from scratch using only retain-class data.

### 2. Fine-Tuning Unlearning (CIFAR-10)

A simpler, more common unlearning approach for a single monolithic model:

- A `SmallCNN` is trained on the **full CIFAR-10 training set** (10 epochs).
- **Unlearning**: the original model's weights are further fine-tuned for a few additional epochs (5) using only the retain-class data (i.e., continuing training but excluding the `airplane` class).
- **Retrained baseline**: a fresh `SmallCNN` is trained from scratch (10 epochs) using only the retain-class data.

## Evaluation

For each experiment and each of the three models (Original / Unlearned / Retrained), the following are measured and visualized:

- **Training / unlearning wall-clock time** (seconds)
- **Overall test accuracy**
- **Per-class accuracy**, with particular attention to the forgotten class's accuracy (the key indicator of successful unlearning — it should drop to ~0%)
- **Confusion matrices** for visual comparison across the three models
- **Training accuracy curves** across epochs (CIFAR-10 experiment)

Results are summarized in formatted tables using `PrettyTable`.

## Models

- **`SimpleCNN`** — a compact convolutional network used for the MNIST/SISA experiment.
- **`SmallCNN`** — a slightly larger convolutional network (with data augmentation) used for the CIFAR-10/fine-tuning experiment.

## Results Summary

### MNIST (SISA) — forgetting digit `7`

| Metric | Original | Unlearned | Retrained |
|---|---|---|---|
| Time (s) | 187.86 | 120.36 | 76.17 |
| Overall Accuracy (%) | 97.90 | 88.09 | 87.42 |
| Forget Class (7) Accuracy (%) | 93.87 | 0.00 | 0.00 |

### CIFAR-10 (Fine-tuning) — forgetting class `airplane`

| Metric | Original | Unlearned | Retrained |
|---|---|---|---|
| Time (s) | 456.30 | 205.41 | 412.88 |
| Overall Accuracy (%) | 75.43 | 71.07 | 67.34 |
| Forget Class (airplane) Accuracy (%) | 74.40 | 0.00 | 0.00 |

**Key takeaways:**
- Both unlearning methods successfully drive the forgotten class's accuracy to 0%, matching the retrained-from-scratch baseline.
- **SISA unlearning** is faster than full retraining in this setup and only slightly slower than the ideal case, since only the shards containing forget-class data need retraining.
- **Fine-tuning unlearning** is substantially faster than retraining from scratch (205s vs. 413s) while retaining higher overall accuracy (71.07% vs. 67.34%), making it an efficient practical alternative when sharded training isn't used.

## Requirements

```
torch
torchvision
scikit-learn
numpy
matplotlib
prettytable
```

Install with:

```bash
pip install torch torchvision scikit-learn numpy matplotlib prettytable
```

Datasets (MNIST, CIFAR-10) are downloaded automatically via `torchvision.datasets` into a local `./data` directory on first run.

## Running the Notebook

1. Install the requirements above.
2. Open `lfdproje.ipynb` in Jupyter or another notebook environment.
3. Run all cells in order:
   - **Cell 1**: model definitions and training utilities.
   - **Cell 2**: MNIST SISA experiment (original / unlearned / retrained + plots).
   - **Cell 3**: CIFAR-10 fine-tuning experiment (original / unlearned / retrained + plots).
   - **Cell 4**: summary tables for both experiments.

> **Note:** Training all six models (three per experiment) is compute-intensive and will take significantly longer on CPU than GPU. A CUDA-capable GPU is recommended.

