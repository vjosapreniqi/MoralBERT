# MoralBERT: A Fine-Tuned Language Model for Capturing Moral Values in Social Discussions

This repository contains the code for the MoralBERT paper accepted at ACM GoodIT 2024 ([click here for the paper](https://dl.acm.org/doi/10.1145/3677525.3678694)). The work involves training BERT models to predict moral values from social media text. 
Baseline models, including a lexicon-based model and a machine learning model, are also provided.

## MoralBERT Web App
If you want to apply moral automatic annotation in your text without having to write any code, head over to [MoralBERTApp](https://huggingface.co/spaces/vjosap/MoralBERTApp).

## Repository Structure

### 1. MoralBERT
This folder contains Python Jupyter Notebook files for training MoralBERT and predicting moral values in text. 
The models are fine-tuned on annotated social media datasets and are designed to understand and predict the representation of moral values based on the Moral Foundations Theory (MFT).
The code demonstrates the prediction of 10 moral foundations, each handled individually by a single classification model. You can also include Liberty/Oppression (when the data is available and annotated accordingly) using the same script.

#### 1.1. Leveraging Deployed Model Weights on Hugging Face for MFT Predictions
The pre-trained weights of our models are now available on Hugging Face, enabling rapid utilisation of the MoralBERT models to compute moral scores for any text and store the results in a DataFrame. To apply these models, please refer to the `MoralBert/Predict_mft_scores_from_the_MoralBERT_weights.ipynb` script. The weights for the liberty/oppression dimension are not yet released, as we are still working on refining this aspect of the model.

### 2. Baselines
This folder hosts Python Jupyter Notebook files for baseline models built with the MoralStrength Lexicon (see the [paper](https://www.sciencedirect.com/science/article/pii/S095070511930526X) and the [GitHub page](https://github.com/oaraque/moral-foundations)) and a Word2Vec with Random Forest Model using [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) machine learning library.

### 3. GPT4
This folder hosts Python Jupyter Notebook file for utilising GPT-4 zero shot classification model for predicting moral foundations.


### 4. Data
To fine-tune the models with the same data, please download the datasets as follows:

- **Moral Foundations Twitter Corpus**: [Download the corpus here](https://osf.io/k5n7y/)
- **Moral Foundations Reddit Corpus**: [Download the corpus here](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
- **Facebook Vaccination Posts**: Please contact the authors of this paper: [Contact Authors](https://dl.acm.org/doi/10.1145/3543507.3583865)
