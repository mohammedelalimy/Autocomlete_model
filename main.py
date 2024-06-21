# Import required libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template


# Define the GPT2 class
class GPT2:
    def __init__(self):
        self.model_type = "GPT2"
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()  # Set the model to evaluation mode

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_next(self, text, k):
        # Encode a text inputs
        indexed_tokens = self.tokenizer.encode(text)
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

        # Get the predicted next sub-word
        probs = predictions[0, -1, :]
        top_next = [self.tokenizer.decode(i.item()).strip() for i in probs.topk(k).indices]

        return top_next


# Initialize the model
gpt2 = GPT2()

# Create the Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    k = data.get('k', 5)  # Default to 5 predictions if not specified
    if not text:
        return jsonify({"error": "Text input is required"}), 400

    try:
        predictions = gpt2.predict_next(text, k)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
