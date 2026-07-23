"""
chatbot.py
-----------
Core FAQ Chatbot logic.
"""

import json
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    cleaned_tokens = [
        LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS and tok.strip()
    ]
    return " ".join(cleaned_tokens)


class FAQChatbot:
    def __init__(self, faq_path: str = "faqs.json"):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        self.processed_questions = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_query: str, threshold: float = 0.25):
        processed_query = preprocess(user_query)
        if not processed_query.strip():
            return "Sorry, I didn't understand that. Could you rephrase your question?", 0.0

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < threshold:
            return (
                "Sorry, I couldn't find a good match for your question. "
                "Please try rephrasing it or contact support.",
                float(best_score),
            )

        return self.answers[best_index], float(best_score)


if __name__ == "__main__":
    bot = FAQChatbot("faqs.json")
    print("FAQ Chatbot (CLI mode) — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        answer, score = bot.get_response(user_input)
        print(f"Bot ({score:.2f}): {answer}\n")
        