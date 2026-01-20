# 🌍 Language Detection Using Recurrent Neural Network (RNN)

## 📌 Project Overview

Language detection is an important Natural Language Processing (NLP) task that involves identifying the language of a given text.  
This project implements a **Recurrent Neural Network (RNN)** model to automatically detect the **language of a sentence** entered by the user.

The system is built as an **end-to-end deep learning application**, including model training, preprocessing, and deployment using **Streamlit**.

---

## 🧠 1. Model Building & Training

- The model is built using **TensorFlow (Keras)**.
- A multilingual text dataset is used for training.
- Text preprocessing includes:
  - Tokenization
  - Sequence padding
  - Label encoding of languages
- A **Simple RNN architecture** is used for sequential text understanding.
- The trained model is saved in **`.h5` format**.
- The tokenizer and label encoder are saved using **pickle**.

### 📁 Relevant Files

- `eda.ipynb` – Data preprocessing, model training & evaluation  
- `SavedModel/simple_rnn_model.h5` – Trained RNN model  
- `SavedModel/tokenizer.pkl` – Tokenizer & label encoder  

---

## 🔍 2. Prediction Pipeline

- User input text is converted into sequences using the saved tokenizer.
- Sequences are padded to a fixed length.
- The trained RNN model predicts probabilities for each language.
- The language with the **highest probability** is selected as the final prediction.
- The model also returns a **confidence score**.

---

## 🌐 3. Streamlit Web Application

An interactive web application is built using **Streamlit**, allowing users to:

- Enter any sentence in any supported language
- Detect the language instantly
- View prediction confidence

### ✨ Features
- Clean and user-friendly UI
- Fast real-time predictions
- Cached model loading for better performance

---

## ▶️ How to Run the Application

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Project4LanguageDetectiionUsingRNN.git
cd Project4LanguageDetectiionUsingRNN
