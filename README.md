# 📩 SMS Spam Classifier  

🌐 **Live Demo**: [Try it here](https://sms-spam-classifier-58no.onrender.com)  

This project builds a machine learning model to classify SMS messages as **spam** or **ham (not spam)**. It demonstrates how **text preprocessing** and **natural language processing (NLP)** techniques can be used to detect spam messages automatically—helping improve communication efficiency and prevent fraud.  

---

## ⚙️ Approach  
- **Data Preprocessing**  
  - Removed stopwords & punctuation  
  - Tokenized text  
  - Performed stemming/lemmatization  
- **Feature Extraction**  
  - Used **TF-IDF Vectorization** to convert text into numerical features  
- **Model Used**  
  - **Multinomial Naive Bayes Classifier** (fast & effective for text classification)  
- **Evaluation Metrics**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

## 🚀 Usage  

### 1. Clone the repository  
```bash
git clone https://github.com/luckypal24112004-coder/SMS_Spam_Classifier.git
cd SMS_Spam_Classifier
