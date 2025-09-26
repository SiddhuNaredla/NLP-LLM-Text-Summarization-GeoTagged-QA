import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

df = pd.read_csv("summaries_geotagged.csv")

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["summary"])

def retrieve_context(question, top_k=1):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, X).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    return " ".join(df.iloc[top_idx]["summary"].values)

def rag_qa(question):
    context = retrieve_context(question, top_k=1)
    input_text = f"question: {question}  context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), context
