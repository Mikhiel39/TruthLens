from flask import Flask, render_template, request, send_file, jsonify
import requests
import redis
import torch
import re
import pickle
import schedule
import time
import threading
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
from fpdf import FPDF

app = Flask(__name__)

# Redis Connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load Pre-trained NLP Model
MODEL_PATH = "misinfo_bert_model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_model():
    """Loads the model from Redis if available, else from local storage."""
    model_pickle = redis_client.get("news_model")
    if model_pickle:
        model = pickle.loads(model_pickle)
    else:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        redis_client.set("news_model", pickle.dumps(model))  # Store in Redis
    return model


model = load_model()

NEWS_API_URL = f"https://saurav.tech/NewsAPI/everything/cnn.json"


def clean_text(text):
    if text:
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^A-Za-z0-9 ]+", "", text)  # Remove special characters
        return text.lower().strip()
    return ""


def classify_news(news_text):
    inputs = tokenizer(news_text, padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction  # 0 = Fake, 1 = Real

# Fetch news articles from API


def fetch_news():
    response = requests.get(NEWS_API_URL)
    if response.status_code == 200:
        news_json = response.json()
        articles = news_json.get("articles", [])
        extracted_data = []
        for article in articles:
            extracted_data.append({
                "author": article.get("author", "Unknown"),
                "url": article["url"],
                "description": article.get("description", ""),
                "content": clean_text(article.get("content", ""))
            })
        return extracted_data
    return []

# Retrain model every 10 minutes


def retrain_model():
    news_articles = fetch_news()
    if not news_articles:
        return
    with open("training_data.txt", "a") as f:
        for article in news_articles:
            label = classify_news(article["content"])
            f.write(f"{article['content']}\t{label}\n")
    redis_client.set("news_model", pickle.dumps(model)
                     )  # Store updated model in Redis


# Schedule retraining
schedule.every(10).minutes.do(retrain_model)


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)


threading.Thread(target=run_scheduler, daemon=True).start()

# Extract news content from URL


def extract_news_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.text for p in paragraphs])
        return clean_text(content)
    except Exception:
        return "Error extracting content"

# Generate a PDF Report


def generate_pdf(credibility, cross_reference, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="News Credibility Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Credibility Score: {credibility}", ln=True)
    pdf.cell(200, 10, txt=f"Cross-Reference: {cross_reference}", ln=True)
    pdf.output(filename)
    return filename

# Home Route


@app.route("/")
def home():
    return render_template("index.html")

# Analyze News Route


@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form["news_url"]
    news_content = extract_news_content(url)
    credibility_score = classify_news(news_content)
    cross_reference = "https://www.bbc.com/news"  # Placeholder
    report_file = generate_pdf(credibility_score, cross_reference)
    return render_template("result.html", score=credibility_score, ref=cross_reference, report=report_file)

# API Endpoint for Automated Testing


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    url = data.get("news_url", "")
    news_content = extract_news_content(url)
    credibility_score = classify_news(news_content)
    return jsonify({
        "credibility_score": credibility_score,
        "cross_reference": "https://www.bbc.com/news"
    })

# Download Report Route


@app.route("/download")
def download():
    return send_file("report.pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
