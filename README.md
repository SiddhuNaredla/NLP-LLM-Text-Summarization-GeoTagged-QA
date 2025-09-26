# NLP-LLM: Text Summarization & Geo-Tagged QA Pipeline

## Objective
Build a system that can:

- Summarize long-form news articles (CNN-DailyMail dataset).  
- Extract geographic locations from the text and geocode them.  
- (Optional) Answer questions from the article using LLM or open-source frameworks.  
- Save output as a CSV file containing:
  - Summarized article
  - Geolocations (latitude, longitude)
  - (Optional) QA response

---

## 📂 Dataset
**CNN-DailyMail News Text Summarization**  

Kaggle Dataset: [Link](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

**Files:**
- `train.csv`
- `test.csv`
- `validation.csv`

**Columns:**
- `article`: Full news article
- `highlights`: Ground-truth human-written summary (for validation)

---

## 🛠️ Tech Stack
- **Transformers**: `t5-small` for text summarization  
- **TF-IDF & RAG-style Retrieval**: For question-answering from generated summaries  
- **SpaCy**: Named Entity Recognition (NER) for extracting GPE (geo-political entities)  
- **Geopy**: Convert extracted locations into latitude/longitude  
- **Pandas**: Data loading and saving CSV files  
- **Flask**: Simple web interface for QA  
- **Google Colab**: GPU-powered execution


---

## 📦 Installation

Install required packages:

```bash
pip install transformers
pip install torch
pip install spacy
pip install geopy
pip install flask
