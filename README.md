# 📰 NewsClassifier

A machine learning-powered web app that classifies news articles into categories based on their headlines and content.

## 🔗 Deployed Link

[NewsClassifier Web App](https://newsclassifier-2311.streamlit.app/)

## 📓 Project Overview

This project is a text classification tool built using a Random Forest classifier. It aims to categorize news articles by analyzing both their **title** and **body content**.

## 🛠️ Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **ML Model:** Random Forest Classifier  
- **Vectorization:** TF-IDF  
- **Deployment:** Streamlit Cloud  

## 📁 Dataset Overview

The dataset used includes:
- **Headline** of the news article  
- **Content/body** of the article  
- **Category/label** of the news  

After preprocessing, the **headline** and **content** were **merged** into a single feature before applying vectorization.

## 🧠 Model Pipeline

1. **Preprocessing**:  
   - Concatenated news headline and body.
   - Cleaned and tokenized the text.
   
2. **Feature Extraction**:  
   - Applied **TF-IDF Vectorization** to convert text into numerical features.

3. **Model Training**:  
   - Trained a **Random Forest Classifier**.
   - Achieved an **F1-score of 0.87** on the validation set.

## 💡 How to Use

1. Visit the [deployed app](https://newsclassifier-2311.streamlit.app/).
2. Enter the **content**, **headline**, or **both together** of a news article.
3. The app will predict and display the **news category**.

## 📎 Project Files

- `app.py` – Streamlit application code.
- `classifier_model_compressed.pkl.gz` – Compressed trained classifier model.
- `vectorizer.pkl` – Saved TF-IDF vectorizer.
- `requirements.txt` – Python dependencies for the project.
- `README.md` – Project documentation.

## 📊 Model Training Notebook

You can find the complete training and preprocessing pipeline in this Kaggle notebook:  
🔗 [Kaggle Notebook Link](https://www.kaggle.com/code/saiswaroop8656/newsclassifier?scriptVersionId=249965923)

---
