<div align="center">

# 🧠 Unsupervised NER Sequence Modeling — Hidden Markov Model with Baum-Welch

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
![Dataset](https://img.shields.io/badge/Dataset-1M%2B%20tokens-blue?style=for-the-badge)
![Sentences](https://img.shields.io/badge/Sentences-47%2C959-orange?style=for-the-badge)

**A from-scratch Hidden Markov Model trained via the Baum-Welch (EM) algorithm on a 1M+ token NER corpus — learning latent linguistic states purely from word sequences, with no label supervision.**

[📓 View Notebook](./project2.ipynb) · [📂 Dataset](./NER%20dataset.csv) · [🐛 Report Bug](https://github.com/ahmedayman2825/sto/issues)

</div>

---

## 📌 Overview

<!-- Add a brief description of your project goals, context, or course here -->
<!-- e.g. "This project was developed as part of the Natural Language Processing / Machine Learning course at Alexandria University ..." -->
<!-- e.g. "The goal was to implement the Baum-Welch algorithm from scratch using only NumPy and apply it to a real NER corpus to discover whether unsupervised HMM states correlate with linguistic categories." -->

This project implements a **Discrete Hidden Markov Model (HMM)** entirely from scratch using NumPy — no HMM libraries — and trains it on a real-world **Named Entity Recognition (NER) corpus** of over **1 million tokens** across **47,959 sentences**.

The model learns **5 latent hidden states** from raw word sequences alone, with no access to POS tags or NER labels during training. The training algorithm is the **Baum-Welch (Forward-Backward EM)** procedure, which iteratively refines three parameter matrices — the initial state distribution **(π)**, the state transition matrix **(A)**, and the emission probability matrix **(B)** — to maximize the likelihood of observing the training sequences.

After training, the learned emission matrix is decoded to reveal which words each latent state has assigned the highest probability to — providing interpretable insight into whether the unsupervised states have discovered meaningful linguistic structure.

---

## ✨ Features

| Feature | Description |
|---|---|
| **From-Scratch HMM** | Full implementation of π, A, B matrices — no `hmmlearn`, no external HMM library |
| **Scaled Forward-Backward** | Numerically stable forward-backward algorithm using per-step scaling coefficients `c[t]` to prevent underflow |
| **Baum-Welch (EM) Training** | Full E-step (γ, ξ accumulators) and M-step (π, A, B re-estimation) across all sequences per iteration |
| **Log-Likelihood Tracking** | Records log P(observations \| model) per iteration to monitor convergence |
| **Vocabulary Building** | Keeps top-K most frequent words (default: 200); all others mapped to `<UNK>` |
| **Subset Training** | Configurable `SUBSET_SIZE` (default: 10,000 sentences) for fast experimentation |
| **Emission Decoding** | Post-training: prints top-20 words per latent state, revealing what each state has learned to represent |
| **Reproducibility** | Fixed `RND_SEED=42` for deterministic parameter initialization |
| **CLI-Ready Structure** | `argparse` imported; `main()` function pattern supports easy command-line execution |

---

## 🗂️ Dataset

**`NER dataset.csv`** — **1,048,575 tokens** across **47,959 sentences**

| Column | Description |
|---|---|
| `Sentence #` | Sentence boundary marker (forward-filled) — e.g. `Sentence: 1` |
| `Word` | Individual token |
| `POS` | Part-of-speech tag (42 unique tags: NNS, NNP, VBD, JJ, ...) — *not used in training* |
| `Tag` | BIO NER label (17 tags) — *not used in training* |

**NER Tag Schema (ground truth — available for evaluation only):**

| Tag | Meaning |
|---|---|
| `O` | Outside any named entity |
| `B-geo` / `I-geo` | Geographic location |
| `B-per` / `I-per` | Person name |
| `B-org` / `I-org` | Organization |
| `B-gpe` / `I-gpe` | Geopolitical entity |
| `B-tim` / `I-tim` | Time expression |
| `B-art` / `I-art` | Artifact |
| `B-eve` / `I-eve` | Event |
| `B-nat` / `I-nat` | Natural phenomenon |

> The HMM is trained **purely on word sequences** — POS tags and NER labels are unused, making this a fully unsupervised learning task.

---

## 🔬 Model Architecture

```
Hidden Markov Model — Discrete, Fully Unsupervised

  Hidden states:    N = 5  (latent linguistic categories)
  Observation vocab: M = 201  (top-200 words + <UNK>)
  Training data:    10,000 sentences (subset of 47,959)
  EM iterations:    3

Parameters learned:
  π  (N,)     — initial state distribution
  A  (N × N)  — state transition probabilities
  B  (N × M)  — emission probabilities per state

Algorithm:
  E-step:  Forward-Backward with scaling (numerically stable)
           → compute γ[t,i] = P(state=i at t | observations)
           → compute ξ[t,i,j] = P(state i→j at t | observations)
  M-step:  Re-estimate π, A, B from accumulated γ and ξ
  Repeat for N_ITERS iterations
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | All matrix operations — forward/backward pass, γ/ξ accumulators, M-step re-estimation, `matrix_power` |
| `pandas` | CSV loading, sentence grouping via `ffill` + `groupby` |
| `collections.Counter` | Vocabulary frequency counting |
| `time` | Per-iteration training time logging |
| `argparse` | CLI argument structure (imported for extensibility) |

> **Zero deep learning dependencies** — no PyTorch, no TensorFlow, no scikit-learn. The entire HMM is implemented in ~100 lines of NumPy.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ahmedayman2825/sto.git
cd sto

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install numpy pandas jupyter
```

### Run the Notebook

```bash
jupyter notebook project2.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

> **Performance note:** Training runs on 10,000 sentences × 3 EM iterations by default. Increasing `SUBSET_SIZE` or `N_ITERS` will significantly increase runtime. The full 47,959-sentence corpus with 20+ iterations may take several minutes on CPU.

---

## ⚙️ Configuration

All hyperparameters are defined at the top of Cell 1 and can be tuned directly:

```python
DATA_PATH    = "NER dataset.csv"   # Path to dataset
ENCODING     = "latin1"            # File encoding
Vocab_size   = 200                 # Top-K words to keep (+ <UNK>)
SUBSET_SIZE  = 10000               # Sentences to train on (None = full corpus)
N_STATES     = 5                   # Number of hidden states
N_ITERS      = 3                   # Baum-Welch EM iterations
VERBOSE      = True                # Print per-iteration progress
RND_SEED     = 42                  # Random seed for reproducibility
```

---

## 📁 Repository Structure

```
sto/
│
├── project.ipynb         # Main notebook — HMM implementation + Baum-Welch training
├── NER dataset.csv        # NER corpus (1,048,575 tokens, 47,959 sentences)
├── .gitignore
└── README.md
```

---

## 📈 Analysis Walkthrough

```
Load NER dataset.csv  (1,048,575 tokens → 47,959 sentences)
    → Forward-fill Sentence # → group words by sentence ID
    → Subset to first 10,000 sentences
    │
    ├── Vocabulary
    │     → Count all word frequencies (Counter)
    │     → Keep top-200 → add <UNK> → M = 201
    │     → Map each word to integer ID
    │
    ├── Baum-Welch Training  (N_STATES=5, N_ITERS=3)
    │     ├── Initialize π, A, B randomly (seed=42)
    │     └── For each iteration:
    │           ├── E-step: scaled forward-backward on every sequence
    │           │     → accumulate γ (state occupancy)
    │           │     → accumulate ξ (transition counts)
    │           ├── M-step: re-estimate π, A, B
    │           └── Log total log-likelihood
    │
    └── Output Learned Parameters
          ├── π  — initial state probabilities (5 values)
          ├── A  — 5×5 transition matrix
          └── B  — top-20 emitted words per state (emission decoding)
```

---

## 🔭 Future Improvements

- [ ] **Increase N_STATES** to 10–17 and compare whether states align with the 17 BIO NER tags
- [ ] **Viterbi decoding** — implement the Viterbi algorithm to assign a most-likely hidden state sequence to each sentence and compare against ground-truth NER tags
- [ ] **Evaluate alignment** — compute cluster purity or V-measure between learned states and ground-truth tags
- [ ] **Expand vocabulary** — test with `Vocab_size = 1000` or `5000` and measure impact on log-likelihood convergence
- [ ] **Full corpus training** — run on all 47,959 sentences with 20+ iterations using batch processing
- [ ] **Visualize emission matrix** — plot B as a heatmap (states × top words) to visually inspect what each state has learned

---

## 👤 Author

 **Ahmed Ayman**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eng-ahmed-ayman/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ahmedayman2825)

**Ahmed Tawfik**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ah-mo-tawfik/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/tAwFiK2005)

**Ashraf Khamis**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ashraf-k-eesa-b8b8802b4)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ashrafeesa)

**Ahmed Abdelwahab**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-abdalwahab/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ahmedabdalwahab)

**Ahmed Emad**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-zamel09/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/AhmedZamel09)

---

<div align="center">
  <sub>If you found this project useful, consider giving it a ⭐</sub>
</div>
