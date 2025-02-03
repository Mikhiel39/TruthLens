TruthLens: Misinformation Detection using AI

TruthLens is an AI-powered web application designed to detect misinformation in news articles. The application uses advanced Natural Language Processing (NLP) models to classify news articles as either real or fake, helping users quickly verify the credibility of the information they consume.

The backend of the application is powered by Flask, which integrates the DeBERTa NLP model to classify the news articles. The frontend is built using HTML, CSS, and JavaScript, providing an intuitive and responsive interface for users to interact with the tool.

Features:
    -News Article Classification: Classifies news headlines as either Fake or Real.
    -Real-time Verification: Fetches top headlines and evaluates their credibility using the AI model.
    -Easy-to-Use Interface: Simple and intuitive user interface built with HTML, CSS, and JavaScript.
    -Cross-checking: Provides a reference URL from a trusted source for verifying the authenticity of the news article.

Tech Stack:
    -Backend: Flask (Python)
    -NLP Model: DeBERTa (Hugging Face Transformers)
    -Frontend: HTML, CSS, JavaScript
    -Model Integration: PyTorch
    -API: News API for fetching current news headlines

Installation:
Prerequisites-

Before running the project, make sure you have the following installed:
    Python 3.x
    pip (Python package installer)
    Virtual Environment (optional, but recommended)

Step-by-Step Guide-

1. Clone the Repository:
      First, clone the repository to your local machine:
      git clone https://github.com/your-username/TruthLens.git
      cd TruthLens

2. Set up a Virtual Environment (optional but recommended):
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install Dependencies:
      Install all required Python dependencies:
      pip install -r requirements.txt

4. Make sure your requirements.txt file includes the following:
      Flask==2.0.1
      torch==1.10.0
      transformers==4.12.0
      requests==2.26.0

5. Download the Pretrained Model:
      The project uses the DeBERTa model for NLP. Download the model from Hugging Face or ensure you have a pre-trained model saved locally, and specify the path in the code if needed.

6. Start the Flask Application:

7. Run the following command to start the Flask application:
      python app.py

How It Works:
    -User Interface: The frontend consists of a simple form where users can input a news headline.
    -Model Prediction: When the user submits the headline, the backend sends the data to the NLP model for classification.
    -Results: The model predicts whether the news article is real or fake, and the result is displayed to the user along with a reference URL.
    -Cross-Verification: If the news is classified as Fake, the application provides a trusted reference URL for further verification.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
    -The DeBERTa model is from Microsoft and implemented using the Hugging Face Transformers library.
    -The News API is used to fetch the latest news headlines for classification.
    -Flask is used for the backend API integration.
