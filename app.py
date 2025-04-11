from flask import Flask, render_template, request, jsonify
from mistral_chatbot import MistralChatbot
import os

app = Flask(__name__)

# Replace with your actual Mistral API key
api_key = "API KEY"
chatbot = MistralChatbot(api_key)

# Load knowledge base from PDFs
pdf_files = ["knowledge_base/knowledge_doc1.pdf", "knowledge_base/knowledge_doc2.pdf"]
chatbot.load_knowledge_base(pdf_files)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Check if the request is JSON
        if not request.is_json:
            return jsonify({"response": "Invalid request format. Please send JSON data."}), 400

        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"response": "Please type a message!"}), 400

        # Debugging: Log the user input
        print(f"User input received: {user_input}")

        response = chatbot.generate_response(user_input)

        # Debugging: Log the generated response
        print(f"Generated response: {response}")

        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing request: {e}")  # Detailed error log
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)