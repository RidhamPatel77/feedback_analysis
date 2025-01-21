import gradio as gr
import joblib

# Load model and vectorizer
SVM = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def sentiment_analysis(input_text):
    input_vector = vectorizer.transform([input_text])
    prediction = SVM.predict(input_vector)
    return f"Sentiment: {prediction[0]}"

# Gradio interface
interface = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(label="Enter Text for Sentiment Analysis"),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Sentiment Analysis with SVM",
    description="Enter text to determine its sentiment using a trained SVM model."
)

interface.launch()
