# Surgical Machine Unlearning via Residual Quantization

Code for the paper "Surgical Machine Unlearning via Residual Quantization" (NeurIPS 2026 submission).

## Overview

This repository implements O(1) exact machine unlearning using frozen foundation model embeddings,
Faiss Residual Quantization, and sufficient statistics tables.

Pipeline: frozen backbone -> Faiss RVQ encoding -> hash table of sufficient statistics -> prediction.
Unlearning = subtract a data point's contribution from the relevant table entries.

## Setup

```bash
pip install -r requirements.txt
```

## Code

(To be added after experiments are implemented.)
