import gradio as gr
import joblib

# Load model and vectorizer
SVM = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizers.pkl")

def sentiment_analysis(input_text):
    if not input_text.strip():  # Handle empty input
        return "Please enter valid text."
    
    # Transform input text using the TfidfVectorizer
    input_vector = vectorizer.transform([input_text])
    
    # Predict sentiment
    prediction = SVM.predict(input_vector)
    if prediction[0]==0:
        return f"The feedback was negative"
    elif prediction[0]==1:
        return f"The feedback was neutral"
    elif prediction[0]==2:
        return f"The feedback was positive"
    else:
        return f"Unknown Error"

# Gradio interface
interface = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(label="Enter Text for Sentiment Analysis"),
    outputs=gr.Textbox(label="Sentiment Result"),
    title="Sentiment Analysis with SVM",
    description="Enter text to determine its sentiment using a trained SVM model."
)

interface.launch()
