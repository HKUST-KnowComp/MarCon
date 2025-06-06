# MarCon
Official repository for our ACL2025 (Main Conference) paper [Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](https://arxiv.org/abs/2505.24778).

## Abstract
As Large Language Models (LLMs) are increasingly used in high-stakes domains, accurately assessing their confidence is crucial. Humans typically express confidence through epistemic markers (e.g., “fairly confident”) instead of numerical values. However, it remains unclear whether LLMs reliably use these markers to reflect their intrinsic confidence due to the
difficulty of quantifying uncertainty associated with various markers. To address this gap, we first define marker confidence as the observed accuracy when a model employs an epistemic
marker. We evaluate its stability across multiple question-answering datasets in both indistribution and out-of-distribution settings for open-source and proprietary LLMs. Our results show that while markers generalize well within the same distribution, their confidence is inconsistent in out-of-distribution scenarios. These findings raise significant concerns about the reliability of epistemic markers for confidence estimation, underscoring the need for improved alignment between marker based confidence and actual model uncertainty. 

## Requirements

conda environments could be setup via:
```
conda env create -f marcon.yml

conda activate marcon
```

clone the repository to local server:
```
git clone https://github.com/HKUST-KnowComp/MarCon.git

cd MarCon
```

## Code Usage
For datasets with binary answer, run:
```
python binary_codespace/prompt_adjustor.py
```
to get the ECE values presented in the paper.

For multiple-choice datasets, run:
```
python MC_codespace/prompt_adjustor.py
```
to get the ECE values presented in the paper.

For the marker analysis experiments (C-AvgCV, MAC, MRC, I-AvgCV), run:
```
python marker_analysis.py
```
to get the results. You can adjust the filtering threshold by changing the value of ```filter_threshold``` in [marker_analysis.py](https://github.com/HKUST-KnowComp/MarCon/blob/main/marker_analysis.py).

## Citing this work
```
@misc{liu2025revisitingepistemicmarkersconfidence,
      title={Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?}, 
      author={Jiayu Liu and Qing Zong and Weiqi Wang and Yangqiu Song},
      year={2025},
      eprint={2505.24778},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.24778}, 
}
```

# Acknowledgement
We thank the anonymous reviewers and the area chair for their constructive comments. The authors of this paper were supported by the ITSP Platform Research Project (ITS/189/23FP) from ITC of Hong Kong, SAR, China, and the AoE (AoE/E-601/24-N), the RIF (R6021-20) and the GRF (16205322) from RGC of Hong Kong SAR, China.

