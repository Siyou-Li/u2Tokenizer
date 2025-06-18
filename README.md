<p>
  <h1>
    <img src="./assets/logo.png" height=150px align="right"/>
   <var>&micro<sup>2</sup></var>Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation
  </h1>
</p>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/pdf/)
[![PWC](https://img.shields.io/badge/%F0%9F%8C%8E%20Website-Official%20Page-blue)]()
[![PWC](https://img.shields.io/badge/HuggingFace-Demo-Green)]()

🎉🎉🎉 Paper accepted by the 28th conference of The Medical Image Computing and Computer Assisted Intervention Society (MICCAI). See you in Daejeon, Korea from September 23-27, 2025.
--

<p align="center">
  <img src="./assets/cover.svg">
</p>


This repository contains the official paper for μ² Tokenizer, a novel approach for automated radiology report generation (RRG) introduced in the paper "μ² Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation".

Our proposed model, μ²LLM, leverages a multi-scale, multi-modal architecture to generate accurate and clinically salient radiology reports from CT scans.

## 🚀 Introduction

<img src="./assets/ullm.svg">

we introduce μ²LLM, a multi-scale multimodal large language model. At its core is the novel μ² Tokenizer, an intermediate layer that intelligently fuses visual features from CT scans with textual information. The model is further refined using Direct Preference Optimization (DPO), guided by the specialized medical report evaluation metric, GREEN, to ensure the generated reports align with expert standards.

<img src="./assets/dpo.svg">

Our experimental results on four large-scale CT datasets show that μ²LLM outperforms existing methods, highlighting its potential for generating high-quality radiology reports even with limited training data.

## ✨ Key Contributions
- μ²LLM Framework: We propose a novel multi-modal large language model (MLLM) designed to efficiently preserve critical details from medical imaging by integrating guided questions.
- μ² Tokenizer Layer: The core of our framework is the μ² Tokenizer, an intermediate layer that uses multi-level attention and multi-scale aggregation to refine and fuse visual and text embeddings, maximizing semantic correspondence while maintaining computational efficiency.
- Enhanced Training with DPO: We employ Direct Preference Optimization (DPO) to align our model's outputs with expert-validated clinical accuracy. The preference data is curated using the GREEN score, a robust LLM-based metric for evaluating the clinical accuracy of radiology reports.


- State-of-the-Art Performance: Despite its smaller parameter size (1B), our model consistently outperforms larger baseline models (7B to 14B) across multiple datasets, demonstrating the effectiveness of our approach.

## 📊 Results

Our model was evaluated on several benchmark datasets against various high-performing LLMs. [cite\_start]We used both traditional metrics (ROUGE, METEOR, BERTScore) and the advanced LLM-based **GREEN** score, which measures clinical accuracy[cite: 99, 100, 101].

### Performance Comparison

[cite\_start]μ²LLM achieves state-of-the-art results across all datasets, significantly outperforming larger models like LaMed-Llama-2-7B and RadFM-14B[cite: 111]. [cite\_start]The use of DPO fine-tuned with GREEN scores further boosted performance, with a notable **20% improvement** in the GREEN score on average[cite: 114, 115].

[cite\_start]**Table 1: Performance Comparison Across Different Datasets** [cite: 97]

| Datasets          | Models                  |  ROUGE-1  |   GREEN   |  METEOR   | BERTScore |
| :---------------- | :---------------------- | :-------: | :-------: | :-------: | :-------: |
|                   | LaMed-Phi-3-4B          |   0.136   |   0.011   |   0.058   |   0.807   |
|                   | LaMed-Llama-2-7B        |   0.139   |   0.009   |   0.060   |   0.810   |
| **Abdomen Atlas** | RadFM-14B               |   0.037   |   0.000   |   0.013   |   0.794   |
|                   | RadGPT-N                |   0.247   |           |   0.112   |           |
|                   | **μ²LLM-1B (SFT)**      | **0.529** | **0.281** | **0.295** | **0.891** |
|                   | **μ²LLM-1B (SFT\&DPO)** | **0.567** | **0.346** | **0.319** | **0.895** |
|                   | LaMed-Phi-3-4B          |   0.130   |   0.002   |   0.050   |   0.814   |
|                   | LaMed-Llama-2-7B        |   0.103   |   0.001   |   0.048   |   0.815   |
| **CT-Rate**       | RadFM-14B               |   0.054   |   0.014   |   0.017   |   0.812   |
|                   | CT-CHAT-8B              |   0.294   |   0.113   |   0.221   |   0.815   |
|                   | **μ²LLM-1B (SFT)**      | **0.517** | **0.384** | **0.330** | **0.879** |
|                   | **μ²LLM-1B (SFT\&DPO)** | **0.539** | **0.429** | **0.359** | **0.890** |
|                   | LaMed-Phi-3-4B          |   0.126   |   0.009   |   0.047   |   0.821   |
|                   | LaMed-Llama-2-7B        |   0.163   |   0.009   |   0.065   |   0.823   |
| **AMOS-MM**       | RadFM-14B               |   0.046   |   0.001   |   0.015   |   0.812   |
|                   | **μ²LLM-1B (SFT)**      | **0.421** | **0.339** | **0.249** | **0.881** |
|                   | **μ²LLM-1B (SFT\&DPO)** | **0.459** | **0.400** | **0.876** | **0.881** |

### Ablation Study

[cite\_start]Ablation experiments confirmed that each component of the μ² Tokenizer contributes positively to the model's performance[cite: 117]. [cite\_start]Differentiable Token Selection (DTS) provided the most significant boost, improving the GREEN score by up to 0.2 points[cite: 118].

[cite\_start]**Table 2: Ablation Study on μ² Tokenizer Components** [cite: 104]

| Model                   |   BLEU    |  ROUGE-1  |   GREEN   |  METEOR   | BERTScore |
| :---------------------- | :-------: | :-------: | :-------: | :-------: | :-------: |
| Baseline                |   0.190   |   0.405   |   0.204   |   0.210   |   0.864   |
| +RPE                    |   0.281   |   0.421   |   0.277   |   0.236   |   0.880   |
| +DTS                    |   0.271   |   0.411   |   0.299   |   0.240   |   0.888   |
| +DMTP                   |   0.254   |   0.401   |   0.233   |   0.220   |   0.874   |
| **μ²LLM-1B (SFT)**      |   0.279   |   0.421   |   0.339   |   0.249   |   0.881   |
| **μ²LLM-1B (SFT\&DPO)** | **0.336** | **0.459** | **0.400** | **0.876** | **0.881** |

-----

## 

## 💿 Datasets

The models were trained and evaluated on the following large-scale CT image-report datasets:

  * [cite\_start]**AMOS-MM 2024** [cite: 80]
  * [cite\_start]**CT-Rate** [cite: 82]
  * [cite\_start]**Abdomen Atlas 3.0** [cite: 84]

[cite\_start]Additionally, we expanded the dataset by using GPT-4o mini to rewrite reports and generate clinically relevant question-answer pairs, enriching the data's diversity[cite: 86, 87].

## 🤖 Model Setup

We use Python 3.10.16 for this project and the library requirements are given in requirements.txt. Create a conda environment using

```
conda create --name <env> --file requirements.txt
```

Ensure that the NVIDIA CUDA version 11.8 or above to be compatible with PyTorch 2.2.2.


The trained checkpoints for our model is available here:
- 


## 🧰 System Hardware requirements

For training, stage 1 and 2 use a 4 * 80GB A100 GPU. For inference, a single 40GB A40 GPU is used. For loading model checkpoint, approximately 39GB of CPU memory is required.

## 🫡 Acknowledgements



## ✨ Cite our work

If you find this repo useful, please consider citing: 

```bibtex

```