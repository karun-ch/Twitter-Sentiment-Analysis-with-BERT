# Twitter Sentiment Analysis with BERT NLP 🐦

## 📌 Project Overview
This project performs **sentiment analysis** on tweets using **BERT (Bidirectional Encoder Representations from Transformers)**. It classifies tweets as **Positive 😊, Neutral 😐, or Negative 😠** using a fine-tuned **BERT model**.

## 🚀 Features
- Uses **Hugging Face's BERT model** (`bert-base-uncased`).
- **Pre-trained on real Twitter data** (`tweet_eval` dataset).
- **Fine-tuned for sentiment classification**.
- **Single notebook implementation** (Google Colab friendly ✅).
- **Predicts sentiment of any given tweet**.

## 📂 Dataset
- The project uses the **tweet_eval sentiment dataset** from **Hugging Face Datasets**.
- It contains labeled tweets as **positive (2), neutral (1), or negative (0)**.

## 🛠️ Installation & Setup
### 1️⃣ Install Dependencies
Run the following command to install required libraries:
```bash
pip install transformers datasets tensorflow
```

### 2️⃣ Load the Dataset
```python
from datasets import load_dataset
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
```

### 3️⃣ Load BERT Tokenizer
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

## 📊 Model Training & Fine-tuning
- The **TFBertForSequenceClassification** model is used for training.
- Uses **SparseCategoricalCrossentropy loss function**.
- Trained for **3 epochs** with **batch size of 16**.

```python
from transformers import TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
```

## 🔍 Predict Sentiment of a Tweet
```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)[0]
    prediction = tf.nn.softmax(outputs, axis=1)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[np.argmax(prediction)]

# Example Usage
print(predict_sentiment("I loved that movie!"))  # Output: Positive ✅
```

## 🖥️ Running on Google Colab
- Upload the notebook to Google Colab.
- Ensure **GPU is enabled** for faster training (`Runtime > Change runtime type > GPU`).

## 🔮 Future Improvements
- **Live Twitter Analysis** (Fetching tweets using Twitter API 🐦).
- **Visualization** (Confusion matrix, word clouds, etc.).
- **Deploy as a Web App** using **Flask or Streamlit**.

---
🚀 **Get Started Now!** Modify the notebook, experiment, and build your own **AI-powered sentiment analysis model!** 🎯

