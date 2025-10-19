from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    return text

@app.route('/')
def home():
    return "✅ Spam Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = preprocess(data['message'])
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    
    return jsonify({'prediction': 'Spam' if prediction[0] == 1 else 'Not Spam'})

# Only use debug locally, not on Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)