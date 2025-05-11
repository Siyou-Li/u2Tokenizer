<p>
  <h1>
    <img src="./assets/logo.png" height=150px align="right"/>
   <var>&micro<sup>2</sup></var>Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation
  </h1>
</p>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/pdf/)
[![PWC](https://img.shields.io/badge/%F0%9F%8C%8E%20Website-Official%20Page-blue)]()
[![PWC](https://img.shields.io/badge/HuggingFace-Demo-Green)]()



<p align="center">
  <img src="./assets/cover.svg">
</p>

## 🚀 Introduction

<img src="./assets/ullm.svg">
we propose $\mu^2$ LLM, a **mu**ltiscale **mu**ltimodal large language models for radiology report generation (RRG) tasks. The novel ${\mu}^2$ Tokenizer, as an intermediate layer, integrates multi-modal features from the multiscale visual tokenizer and the text tokenizer, then enhances report generation quality through direct preference optimization (DPO), guided by GREEN-RedLlama. Experimental results on four large CT image-report medical datasets demonstrate that our method outperforms existing approaches, highlighting the potential of our fine-tuned $\mu^2$ LLMs on limited data for RRG tasks.

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