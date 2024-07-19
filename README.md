# MoralBERT: A Fine-Tuned Language Model for Capturing Moral Values in Social Discussions

This repository contains the code for the MoralBERT paper accepted at ACM GoodIT 2024. The work involves training BERT models to predict moral values from social media text. 
Baseline models, including a lexicon-based model and a machine learning model, are also provided.

## Repository Structure

### 1. MoralBERT
This folder contains Python Jupyter Notebook files for training MoralBERT and predicting moral values in text. 
The models are fine-tuned on annotated social media datasets and are designed to understand and predict the representation of moral values based on the Moral Foundations Theory (MFT).
The code demonstrates the prediction of 10 moral foundations, each handled individually by a single classification model. You can also include Liberty/Oppression (when the data is available and annotated accordingly) using the same script.

### 2. Baselines
This folder hosts Python Jupyter Notebook files for baseline models built with the MoralStrength Lexicon and a Word2Vec with Random Forest Model.

### 3. GPT4
This folder hosts Python Jupyter Notebook file for utilising GPT-4 zero shot classification model for predicting moral foundations.


### 4. Data
To fine-tune the models with the same data, please download the datasets as follows:

- **Moral Foundations Twitter Corpus**: [Download the corpus here](https://osf.io/k5n7y/)
- **Moral Foundations Reddit Corpus**: [Download the corpus here](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
- **Facebook Vaccination Posts**: Please contact the authors of this paper: [Contact Authors](https://dl.acm.org/doi/10.1145/3543507.3583865)
