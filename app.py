from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot("faqs.json")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_message = data.get("message", "")
    answer, score = bot.get_response(user_message)
    return jsonify({"answer": answer, "confidence": round(score, 2)})


if __name__ == "__main__":
    app.run(debug=True)