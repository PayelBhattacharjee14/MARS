# MARS: Margin-Aware Reward Modeling with Self-Refinement

**MARS** is an adaptive reward modeling framework that focuses on data augmentation and refinement on *ambiguous (low-margin) preference comparisons*. By explicitly leveraging reward-model uncertainty (margins), MARS improves curvature, conditioning, and downstream alignment performance compared to uniform or margin-agnostic augmentation strategies.

This repository contains the full codebase for training reward models, aligning policies, and evaluating alignment quality across multiple preference datasets.

---

## Overview

Modern alignment pipelines (e.g., RLHF, RLAIF) rely heavily on high-quality reward models trained from pairwise human preferences. However, existing data augmentation methods typically expand datasets uniformly, ignoring where the reward model struggles most.

**MARS** addresses this limitation by:
- Identifying *low-margin* (ambiguous) preference pairs,
- Concentrating augmentation and refinement on these hard examples,
- Iteratively improving reward model curvature and alignment performance.

We evaluate MARS on multiple widely used alignment benchmarks and show consistent gains in reward modeling accuracy and policy alignment.

