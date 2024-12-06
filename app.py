import re
import fitz
import joblib
import tempfile
import requests
import gradio as gr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the saved model, labels and vectorizer
model = joblib.load("models/product_classifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
labels = joblib.load("models/labels.pkl")


# Helper function to download and extract text from a PDF
def extract_text_from_pdf_url(pdf_url):
    """
    Downloads a PDF from the given URL and extracts text from it using a temporary file.
    """
    try:
        # Download the file
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Use a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf.flush()  # Ensure all data is written before processing

            # Extract text using PyMuPDF
            with fitz.open(temp_pdf.name) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()

        return text
    except Exception as e:
        return f"Error processing PDF: {e}"


def preprocess_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\d+", " ", text)  # Remove numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return " ".join(tokens)


# Prediction function
def predict_pdf_label(pdf_url):
    """
    Predicts the class of a PDF file from its URL and returns probabilities.
    """
    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf_url(pdf_url)
    if text.startswith("Error"):
        return text, {}

    # Step 2: Preprocess the text
    clean_text = preprocess_text(text)

    # Step 3: Convert text into TF-IDF features
    features = vectorizer.transform([clean_text])

    # Step 4: Predict the class and probabilities
    predicted_label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Step 5: Format the probabilities for output
    prob_dict = {labels[i]: round(prob, 4) for i, prob in enumerate(probabilities)}

    return predicted_label, prob_dict


# Gradio app
demo = gr.Interface(
    fn=predict_pdf_label,
    inputs=gr.Textbox(
        label="Enter PDF URL",
        placeholder="Paste the URL of the PDF here...",
    ),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.JSON(label="Class Probabilities"),
    ],
    title="PDF Classifier",
    description="Upload a PDF URL to classify it into one of the 4 categories: Lighting, Fuses, Cables, or Others.",
    flagging_mode="never",
)

# Launch the app
demo.launch()
