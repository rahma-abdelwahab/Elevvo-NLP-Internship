# Natural Language Processing Tasks – Elevvo Pathways Internship

Welcome to the **Natural Language Processing (NLP) Projects Repository** created as part of my internship at **Elevvo Pathways**.

---

## Internship Overview

As part of the Elevvo Internship Program, I completed a series of **NLP-focused projects** that demonstrate foundational to intermediate-level natural language processing techniques.

Each task involved:

* Working with **real-world datasets**.
* Applying **text preprocessing techniques**.
* Implementing **machine learning and deep learning models**.
* Evaluating results using **standard metrics**.

This repository contains **6 NLP tasks** organized across different domains — covering topics from **sentiment analysis to question answering with transformers**.

---

## Repository Structure

```
├── Task_1_Sentiment_Analysis_on_Product_Reviews.ipynb  
├── Task_2_News_Category_Classification.ipynb  
├── Task_3_Fake_News_Detection.ipynb  
├── Task_4_Task_4_Named_Entity_Recognition_(NER).ipynb  
├── Task_5_Topic_Modeling_on_News_Articles.ipynb  
├── Task_6_Question_Answering_with_Transformers.ipynb  
└── README.md  
```

---

## Task Details

###  Task 1: Sentiment Analysis on Product Reviews

* **Dataset**: Amazon Product Reviews / IMDb Reviews (Kaggle).
* **Objective**: Classify customer reviews as **positive, negative, or neutral**.
* **Steps**:

  * Text preprocessing (lowercasing, stopword removal, tokenization, stemming, lemmatization).
  * Text representation using **Bag-of-Words** and **TF-IDF**.
  * Trained classifiers: **Logistic Regression, Naive Bayes, SVM**.
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.
* **Bonus**:

  * WordClouds for **positive and negative reviews**.
  * Comparison of classifier performance.

---

###  Task 2: News Category Classification

* **Dataset**: AG News dataset.
* **Objective**: Classify news articles into **World, Sports, Business, and Sci/Tech** categories.
* **Steps**:

  * Preprocessing: lowercasing, stopword removal, punctuation removal.
  * Feature extraction using **TF-IDF vectorization**.
  * Models trained: **Logistic Regression, Naive Bayes, SVM**.
* **Evaluation Metrics**: Classification Report, Confusion Matrix, Accuracy.
* **Bonus**: Identified **most informative features per class** using feature weights.

---

###  Task 3: Fake News Detection

* **Dataset**: Kaggle Fake News dataset.
* **Objective**: Detect whether a given news article is **real or fake**.
* **Steps**:

  * Preprocessing: lowercasing, stopword removal, stemming, lemmatization.
  * Feature representation: **CountVectorizer** and **TF-IDF**.
  * Models: **Logistic Regression, Random Forest, SVM**.
* **Evaluation Metrics**: Accuracy, F1-score, ROC-AUC, Confusion Matrix.
* **Bonus**: Visualized **distribution of real vs fake articles** and feature importance.

---

###  Task 4: Named Entity Recognition (NER)

* **Dataset**: CoNLL-2003 English Dataset.
* **Objective**: Identify entities such as **Person, Organization, Location, Date**.
* **Steps**:

  * Implemented a **rule-based NER** approach.
  * Applied **spaCy models** (`en_core_web_sm` and `en_core_web_md`).
  * Compared entity extraction between small and medium models.
  * Visualized entities with **spaCy displaCy**.
* **Evaluation**: Qualitative comparison of extracted entities across models.
* **Bonus**: Experimented with **custom regex-based NER rules**.

---

###  Task 5: Topic Modeling on News Articles

* **Dataset**: BBC News Articles dataset.
* **Objective**: Discover hidden **topics** within the articles.
* **Steps**:

  * Preprocessing: lowercasing, stopword removal, lemmatization.
  * Applied **CountVectorizer** and **TF-IDF**.
  * Topic Modeling techniques:

    * **Latent Dirichlet Allocation (LDA)**.
    * **Non-negative Matrix Factorization (NMF)**.
* **Evaluation**: Interpreted topics by analyzing top words per cluster.
* **Bonus**: Visualized topics with **WordClouds**.

---

###  Task 6: Question Answering with Transformers

* **Dataset**: **SQuAD v1.1 (Stanford Question Answering Dataset)**.
* **Objective**: Build a **Question Answering (QA) system** using Transformer-based models.
* **Steps**:

  * Preprocessing: Load contexts, questions, and answers from JSON format.
  * Models evaluated:

    * DistilBERT (`distilbert-base-uncased-distilled-squad`)
    * BERT (`bert-large-uncased-whole-word-masking-finetuned-squad`)
    * RoBERTa (`deepset/roberta-base-squad2`)
    * ALBERT (`twmkn9/albert-base-v2-squad2`)
  * Implemented **Hugging Face pipelines** for QA.
  * Built an **interactive interface** (CLI + optional Streamlit).
* **Evaluation Metrics**:

  * **Exact Match (EM)**
  * **F1 Score**
* **Bonus**: Model performance comparison + live interactive QA system.

