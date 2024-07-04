# MoralBERT: A Fine-Tuned Language Model for Capturing Moral Values in Social Discussions

This repository contains the code for the MoralBERT paper submitted to ACM GoodIT 2024. The work involves training BERT models to predict moral values from social media text. 
Baseline models, including a lexicon-based model and a machine learning model, are also provided.

## Repository Structure

### 1. MoralBERT_Code
This folder contains Python Jupyter Notebook files for training MoralBERT and predicting moral values in text. 
The models are fine-tuned on annotated social media datasets and are designed to understand and predict the representation of moral values based on the Moral Foundations Theory (MFT).

### 2. Baseline_Code
This folder hosts Python Jupyter Notebook files for baseline models built with the MoralStrength Lexicon and a Word2Vec with Random Forest Model.

### 3. Data
To fine-tune the models with the same data, please download the datasets as follows:

- **Moral Foundations Twitter Corpus**: [Download the corpus here](https://osf.io/k5n7y/)
- **Moral Foundations Reddit Corpus**: [Download the corpus here](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
- **Facebook Vaccination Posts**: Please contact the authors of this paper: [Contact Authors](https://dl.acm.org/doi/10.1145/3543507.3583865)
