## Progressive distillation induces an implicit curriculum

This repository contains the code for our paper [Progressive distillation induces an implicit curriculum](https://openreview.net/forum?id=wPMRwmytZe)

## Quick Links
- [Overview](#overview)
- [Main Results](#main-results)
- [Experiments](#experiments)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview
Knowledge distillation leverages a teacher model to improve the training of a student model. A persistent challenge is that a better teacher does not always yield a better student, to which a common mitigation is to use additional supervision from several “intermediate” teachers. One empirically validated variant of this principle is progressive distillation, where the student learns from successive intermediate checkpoints of the teacher. Using sparse parity as a sandbox, we identify an implicit curriculum as one mechanism through which progressive distillation accelerates the student’s learning. This curriculum is available only through the intermediate checkpoints but not the final converged one, and imparts both empirical acceleration and a provable sample complexity benefit to the student. We then extend our investigation to Transformers trained on probabilistic context-free grammars (PCFGs) and real-world pre-training datasets (Wikipedia and Books). Through probing the teacher model, we identify an analogous implicit curriculum where the model progressively learns features that capture longer context. Our theoretical and empirical findings on sparse parity, complemented by empirical observations on more complex tasks, highlight the benefit of progressive distillation via implicit curriculum across setups.

## Main Results


## Experiments


## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Simon (juhyunp 'at' princeton 'dot' edu), Abhishek (ap34 'at' princeton 'dot' edu), and Yun (yc6206 'at' princeton 'dot' edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give more effective help!

## Citation

Please cite our paper if you find our paper or this repo helpful:
```bibtex
@misc{panigrahi2025progressive,
title={Progressive distillation induces an implicit curriculum},
author={Abhishek Panigrahi and Bingbin Liu and Sadhika Malladi and Andrej Risteski and Surbhi Goel},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=wPMRwmytZe}
}
```
