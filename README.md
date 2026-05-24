# MARS: Margin and Semantic-Aware Data Augmentation for Reward Modeling

**MARS** is a margin and semantic-aware data augmentation framework for reward modeling. Instead of uniformly expanding preference datasets, MARS uses the reward model's current margins to identify uncertain or mis-ranked preference pairs and concentrates synthetic augmentation on those examples. It further uses semantic-distance analysis to decide whether selected chosen-rejected pairs should be directly augmented or first refined to preserve a clear preference signal.

This repository contains the codebase for reward model training, semantic-distance analysis, preference augmentation, policy alignment, and evaluation across multiple preference datasets and alignment benchmarks.

## Overview

Modern alignment pipelines such as RLHF, RLAIF, and PPO-based policy optimization rely heavily on reward models trained from pairwise preference data. However, high-quality preference annotations are expensive to collect, and standard augmentation strategies often expand all preference pairs uniformly without considering where the reward model is uncertain or prone to misranking.

MARS addresses this limitation through two coupled components:

- **Margin-aware augmentation:** MARS computes reward margins for preference pairs and allocates more augmentation budget to low-margin or mis-ranked examples where the reward model needs stronger corrective supervision.
- **Semantic-aware refinement:** MARS measures the semantic distance between chosen and rejected responses. Semantically well-separated pairs are augmented directly, while semantically close pairs are first rewritten to sharpen the chosen-rejected contrast before augmentation.

By combining margin-based allocation with semantic refinement, MARS targets supervision where it is most informative while avoiding redundant synthetic variants with weak ranking signal.

## Experiments

We evaluate MARS across multiple preference datasets, reward-model backbones, and downstream alignment settings.

**Datasets**
- HH-RLHF
- UltraFeedback
- PKU-SafeRLHF

**Reward-model backbones**
- DeBERTa-v3-base
- RoBERTa-base
- RoBERTa-large

**Aligned policy models**
- Llama-3.2-1B
- TinyLlama-1.1B

**Evaluation benchmarks**
- RewardBench
- AlpacaEval

Across these settings, MARS improves reward-model quality on RewardBench and leads to stronger downstream alignment performance on RewardBench and AlpacaEval compared to baselines such as uniform augmentation, no augmentation, AdaBoost-style reweighting, and reward-model self-training approaches.

## Repository Structure

The repository is organized around the main stages of the \texttt{MARS} pipeline:

.
├── LICENSE
├── README.md
├── semantic_and_paraphrasing.ipynb
├── RM_training.ipynb
├── RM_evaluation.ipynb
├── alignment.ipynb
└── alignment_evaluation.ipynb


## Code

The repository includes the full experimental pipeline used in the paper, including scripts for:

1. fixed 1k preference subset construction;
2. semantic-distance computation and high/low semantic labeling;
3. GPT-based semantic refinement for low-distance pairs;
4. paraphrase-based synthetic preference generation;
5. reward model training with MARS and baselines;
6. PPO-style policy alignment using trained reward models;
7. RewardBench and AlpacaEval evaluation;
8. plotting and qualitative generation analysis.

## Citation

Citation information will be added after publication.
