# CS310 NLP Project: Human vs. Machine-Generated Text Detection

## Overview

This project compares **supervised learning** (text-based and spectrum-based) and **zero-shot detection** methods to distinguish human-written from machine-generated text. Evaluations are conducted on English and Chinese datasets for in-domain and cross-domain performance.

**Authors**: Lou Yibin (12310513), Rui Yuhan (12310520)

## Results
### Three Method in domain
<p align="center">
  <img src="images/Three Method in domain Accuracy(Eng).png" alt="Result 1" width="300"/>
  <img src="images/Three Method in domain Accuracy(Zh).png" alt="Result 2" width="300"/>
</p>


### Three Method out of domain
<p align="center">
  <img src="images/Three Method out of domain Accuracy(Eng).png" alt="Result 1" width="300"/>
  <img src="images/Three Method out of domain Accuracy(Zh).png" alt="Result 2" width="300"/>
</p>

### Extended Evaluation (Mixed Domain & One-Shot)
<!-- 第一行 -->
<p align="center">
  <img src="images/Overal evaluation in mixed domain(Eng) nll.png" alt="Result 1" width="200"/>
  <img src="images/Overal evaluation in mixed domain(Eng).png" alt="Result 2" width="200"/>
</p>

<!-- 第二行 -->
<p align="center">
  <img src="images/Overal evaluation in mixed domain(Zh) nll.png" alt="Result 3" width="200"/>
  <img src="images/Overal evaluation in mixed domain(Zh).png" alt="Result 4" width="200"/>
</p>

<!-- 第三行 -->
<p align="center">
  <img src="images/Overal evaluation in one-shot models(Eng).png" alt="Result 5" width="200"/>
  <img src="images/Overal evaluation in one-shot models(Zh).png" alt="Result 6" width="200"/>
</p>

## Datasets

- **English**:
  - Source: Ghostbuster dataset ([GitHub](https://github.com/vivek3141/ghostbuster-data))
  - Domains: Essay, Reuter, WP
  - Format: CSV (text, label: 1 for LLM, 0 for HM, domain); split 8:1:1 (train/validation/test)
  - Zero-shot: Separate HM/LLM text files
- **Chinese**:
  - Domains: News, Webnovel, Wiki
  - Content: Human-written and Qwen2-72b-generated texts
  - Preprocessing: Same as English; input-only format performs better

## Methods

- **Supervised Learning (Text-Based)**:
  - Fine-tune Transformer models (e.g., `roberta-base`, `bert-base-chinese`) for binary classification.
  - Metrics: Accuracy, precision, recall, F1, AUROC
- **Supervised Learning (Spectrum-Based)**:
  - Use spectral features from NLL scores (models: `gpt2-xl`, `Wenzhong2.0-GPT2-3.5B-chinese`).
  - Train SVM classifier with RBF kernel.
- **Zero-Shot Detection**:
  - Heuristic spectrum-based classifier using low-frequency features and threshold \(\varepsilon\).

## Results

- **Supervised (Text-Based)**:
  - English: `roberta-base` best in mixed-domain (>0.98 accuracy); in-domain >0.88, out-of-domain ~0.80
  - Chinese: `xlm-roberta-base` leads in AUROC/F1; in-domain >0.83, out-of-domain 0.63–0.72
- **Supervised (Spectrum-Based)**:
  - English: `Mistral-7B-v0.1` slightly better; in-domain >0.95, out-of-domain <0.6
  - Chinese: `Wenzhong2.0-GPT2-3.5B` outperforms; in-domain >0.8, out-of-domain <0.6
- **Zero-Shot**:
  - English: `Mistral-7B-v0.1` achieves >0.75 accuracy in Essay/Reuter, ~0.59 in WP
  - Chinese: `Wenzhong2.0-GPT2-3.5B` best in Webnovel (0.73), weakest in News (0.64)

## Installation

- **Requirements**: Python 3.8+, `torch`, `transformers`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`

- Install:

  ```bash
  pip install torch transformers numpy scipy scikit-learn pandas matplotlib
  ```

## Usage

1. **Prepare Data**:

   - Download Ghostbuster dataset for English.
   - Place Chinese data in appropriate folder.
   - Run preprocessing scripts to generate CSV/TXT files.

2. **Run Experiments**:

   - Supervised (Text): `python supervised_text.py --model roberta-base --data eng_mix.csv`
   - Supervised (Spectrum): `python supervised_spectrum.py --model gpt2-xl --data eng_essay_llm.txt`
   - Zero-Shot: `python zero_shot.py --model Mistral-7B-v0.1 --data eng_essay_hm.txt eng_essay_llm.txt`

3. **Compile Report**:

   ```bash
   cd report
   latexmk -pdf main.tex
   ```

## Conclusion

- **Supervised learning** excels in-domain; text-based outperforms spectrum-based.
- **Zero-shot** offers better out-of-domain generalization, especially for Chinese.
- **Future Work**: Explore larger models, multi-modal data, and advanced spectral techniques.

## Citation

```bib
@misc{cs310_nlp_2025,
  author = {Lou, Yibin and Rui, Yuhan},
  title = {CS310 NLP Project: Detecting Human vs. Machine-Generated Text},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Nahuyiur/CS310-NLP}}
}
```

## Contact

- Lou Yibin: 12310513@mail.sustech.edu.cn
- Rui Yuhan: 12310520@mail.sustech.edu.cn
