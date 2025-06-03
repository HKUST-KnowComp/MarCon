# MarCon
Official repository for our ACL2025 (Main Conference) paper [Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](https://arxiv.org/abs/2505.24778).

## Introduction
This paper investigates whether large language models can reliably/consistently express their confidence using epistemic markers instead of numerical values. Our findings indicate that while LLMs' in-distribution marker confidence is relatively stable, its **consistency declines in out-of-distribution scenarios in different perspectives**, raising concerns about the reliability of such markers for confidence estimation.

## Requirements

```
git clone https://github.com/HKUST-KnowComp/MarCon.git

cd MarCon
```

## Conda Environment

```
conda env create -f marcon.yml

conda activate marcon
```

## Code Usage
[TBD]

## Citing this work
[TBD]

# Acknowledgement
This paper investigates whether large language models (LLMs) can reliably express their confidence using epistemic markers (e.g., "fairly certain") instead of numerical values. The findings indicate that while LLMs' in-distribution marker confidence is relatively stable, its consistency declines in out-of-distribution scenarios, raising concerns about the reliability of such markers for confidence estimation.
