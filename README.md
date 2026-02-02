# 📰 FactOrFake: Automatic Fact Checking Using Machine Learning Models

## 📌 Project Overview
With the rapid growth of social media and online news platforms, fake news spreads very quickly and can cause serious social, political, and economic impacts.  
This project, **FactOrFake**, aims to automatically classify news articles as **Fake** or **Real** using **Machine Learning** and **Natural Language Processing (NLP)** techniques.

---

## 🎯 Problem Statement
Manual verification of news articles is time-consuming and inefficient.  
The objective of this project is to build an automated system that can detect fake news based on textual content using machine learning models.

---

## 📂 Dataset Description
The datasets used in this project were collected from **Kaggle**:

- **Fake and Real News Dataset**  
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Dataset Files
- `Fake.csv` → Fake news articles (Label: `0`)
- `True.csv` → Real news articles (Label: `1`)

---

## ⚙️ Methodology

### Step 1: Data Collection
- Collected fake and real news datasets using Kaggle
- Used **Google Colab** for handling large datasets efficiently

---

### Step 2: Data Preparation
- Loaded datasets using Pandas
- Added labels:
  - Fake news → `0`
  - Real news → `1`
- Combined both datasets into a single dataset
- Shuffled the data to remove bias

---

### Step 3: Text Preprocessing
- Converted text to lowercase
- Removed URLs, punctuation, numbers, and special characters
- Removed stopwords using NLTK
- Cleaned text to improve model performance

---

### Step 4: Feature Extraction
- Applied **TF-IDF Vectorization** to convert text into numerical features
- Reduced noise and focused on important words
- Split data into training and testing sets

---

### Step 5: Machine Learning Models
The following models were trained and evaluated:
- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machine (SVM)

---

### Step 6: Evaluation & Visualization
- Generated **Confusion Matrices**
- Calculated:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Visualized results using heatmaps and bar charts

---

## 📊 Results & Analysis
- Logistic Regression and SVM showed higher accuracy compared to Naive Bayes
- Naive Bayes was faster but less accurate due to independence assumptions
- SVM handled high-dimensional TF-IDF features effectively

**Best Performing Models:**  
✔ Logistic Regression  
✔ Support Vector Machine (SVM)

---

## 🧠 Conclusion
The project successfully demonstrates that machine learning models combined with NLP techniques can effectively detect fake news. Logistic Regression and SVM performed better than Naive Bayes in terms of overall accuracy and classification metrics.

---

## 🚀 Future Work
- Implement deep learning models such as LSTM or BERT
- Perform real-time fake news detection
- Include multilingual news datasets
- Deploy the model as a web application

---

## 🛠️ Technologies Used
- Python
- Google Colab
- Pandas, NumPy
- NLTK
- Scikit-learn
- Matplotlib, Seaborn

---

## 📚 References
- Kaggle Datasets
- Research Papers on Fake News Detection
- Scikit-learn Documentation
