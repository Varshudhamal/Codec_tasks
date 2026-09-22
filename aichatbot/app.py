from flask import Flask, render_template, request, jsonify
import sqlite3
import random
import re
import nltk

from transformers import pipeline

app = Flask(__name__)


# CONFIGURATION


DB_NAME = "chatbot.db"

# Store recent conversation for contextual responses
conversation_history = []


# NLTK SETUP

def setup_nltk():
    """
    Download required NLTK tokenizer data if it is missing.
    """

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download("punkt", quiet=True)


setup_nltk()



# NLTK TEXT PREPROCESSING


def preprocess_text(text):
    """
    Uses NLTK to tokenize and clean user input.
    """

    text = text.lower().strip()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Keep useful words
    tokens = [
        token for token in tokens
        if re.match(r"[a-zA-Z0-9]+", token)
    ]

    return tokens


# SQLITE DATABASE

def create_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_chat(user_message, bot_response):

    try:

        conn = sqlite3.connect(DB_NAME)

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_logs
            (user_message, bot_response)
            VALUES (?, ?)
            """,
            (user_message, bot_response)
        )

        conn.commit()
        conn.close()

    except sqlite3.Error as error:

        print("Database error:", error)


# FAQ DATA


faq = {

    "hello": [
        "Hello! Welcome to our online course support. How can I help you?",
        "Hi! How can I assist you today?"
    ],

    "hi": [
        "Hi! Welcome to our online course support.",
        "Hello! What can I help you with?"
    ],

    "help": [
        "I can help you with courses, enrollment, fees, payments, classes, study materials, assignments, exams, certificates, refunds and technical problems."
    ],

    "course": [
        "We offer courses in Python, Web Development, Data Science, Artificial Intelligence and Machine Learning."
    ],

    "courses": [
        "You can explore available courses from the Courses section."
    ],

    "course information": [
        "Course information such as syllabus, duration, fees and requirements is available on the course details page."
    ],

    "course duration": [
        "Course duration depends on the selected course. Please check the course details page."
    ],

    "course syllabus": [
        "The complete syllabus is available on the respective course details page."
    ],

    "course requirements": [
        "Course requirements depend on the selected course. Basic computer knowledge may be required for some courses."
    ],

    "python": [
        "The Python course covers Python basics, functions, object-oriented programming, file handling and practical projects."
    ],

    "web development": [
        "The Web Development course covers HTML, CSS, JavaScript and web application development."
    ],

    "data science": [
        "The Data Science course covers Python, NumPy, Pandas, data visualization and machine learning."
    ],

    "machine learning": [
        "The Machine Learning course covers supervised learning, unsupervised learning and practical machine learning projects."
    ],

    "ai": [
        "The AI course introduces artificial intelligence concepts, machine learning and practical applications."
    ],

    "enroll": [
        "To enroll, select your desired course and click the Enroll Now button."
    ],

    "enrollment": [
        "You can enroll in a course by selecting the course and completing the registration and payment process."
    ],

    "how to enroll": [
        "Choose a course, click Enroll Now, create or log in to your account and complete the payment."
    ],

    "enrollment problem": [
        "If you are having trouble enrolling, please check your account and payment details or contact customer support."
    ],

    "join course": [
        "After successful enrollment, you can access the course from your student dashboard."
    ],

    "fees": [
        "Course fees depend on the selected course. Please check the course details page for the current fee."
    ],

    "fee": [
        "The course fee is displayed on the course details page."
    ],

    "course price": [
        "You can find the current course price on the course details page."
    ],

    "discount": [
        "Discounts may be available for selected courses. Please check the current offers."
    ],

    "offer": [
        "Please check the Offers section for current discounts and special promotions."
    ],

    "payment": [
        "Available payment methods are displayed on the checkout page."
    ],

    "payment methods": [
        "We support the online payment methods available during checkout."
    ],

    "payment failed": [
        "If your payment failed, please check your payment details and try again."
    ],

    "money deducted": [
        "If money was deducted but enrollment was not confirmed, please wait for the payment status to update or contact support."
    ],

    "payment receipt": [
        "Your payment receipt may be available in your account or sent to your registered email."
    ],

    "classes": [
        "You can access your enrolled classes from the student dashboard."
    ],

    "class": [
        "Your course classes are available in the course section after enrollment."
    ],

    "live class": [
        "If the course includes live classes, the schedule and joining link will be available in your student dashboard."
    ],

    "class schedule": [
        "The class schedule is available in the course dashboard."
    ],

    "missed class": [
        "If recorded sessions are available, you can watch the missed class from the course dashboard."
    ],

    "video": [
        "Course videos can be accessed from your enrolled course dashboard."
    ],

    "video not working": [
        "Please check your internet connection, refresh the page and try playing the video again."
    ],

    "video not playing": [
        "Try refreshing the page, checking your internet connection or using another browser."
    ],

    "video download": [
        "Video download availability depends on the course. Please check the course settings."
    ],

    "study material": [
        "Study materials are available in the course dashboard for enrolled students."
    ],

    "notes": [
        "Course notes and learning resources can be accessed from the Study Material section."
    ],

    "pdf": [
        "If PDF notes are provided, you can download them from the Study Material section."
    ],

    "download material": [
        "Open your course dashboard and go to the Study Material section to download available resources."
    ],

    "assignment": [
        "Assignments are available in the Assignments section of your course dashboard."
    ],

    "assignments": [
        "You can view and submit your assignments from the course dashboard."
    ],

    "submit assignment": [
        "Open the assignment, upload your required file and click the Submit button."
    ],

    "assignment deadline": [
        "The assignment deadline is displayed with the respective assignment."
    ],

    "assignment problem": [
        "If you have a problem submitting an assignment, please check the file format and internet connection."
    ],

    "quiz": [
        "Quizzes are available in the Quiz section of your enrolled course."
    ],

    "exam": [
        "Exam information and schedules are available in your course dashboard."
    ],

    "exam date": [
        "Please check the Exams section of your student dashboard for the latest exam schedule."
    ],

    "quiz problem": [
        "If you experience a problem during a quiz, refresh the page and check your internet connection."
    ],

    "result": [
        "Your quiz or exam results can be viewed from the Results section when they are published."
    ],

    "certificate": [
        "You can receive your course certificate after completing the required course activities."
    ],

    "certificate download": [
        "After completing the course requirements, you can download your certificate from your student dashboard."
    ],

    "certificate not received": [
        "Please check whether you have completed all course requirements. If the certificate is still unavailable, contact support."
    ],

    "certificate correction": [
        "For certificate corrections, please contact customer support with your correct details."
    ],

    "certificate name": [
        "Please make sure your name is correct in your account before the certificate is generated."
    ],

    "account": [
        "You can manage your profile, courses and account information from your student dashboard."
    ],

    "register": [
        "Click the Sign Up or Register option and enter your required details to create an account."
    ],

    "registration": [
        "You can register by creating an account with your email and required information."
    ],

    "login": [
        "Enter your registered email and password to log in to your account."
    ],

    "forgot password": [
        "Click the Forgot Password option on the login page and follow the instructions to reset your password."
    ],

    "change password": [
        "You can change your password from your account settings."
    ],

    "cannot login": [
        "Please check your email and password. If the problem continues, use Forgot Password or contact support."
    ],

    "technical support": [
        "Our technical support team can help with login, videos, website and course-access problems."
    ],

    "website not working": [
        "Please check your internet connection and refresh the website. If the problem continues, contact technical support."
    ],

    "page not loading": [
        "Please refresh the page and check your internet connection. You can also try another browser."
    ],

    "course not showing": [
        "Please check your student dashboard. If your enrolled course is still missing, contact customer support."
    ],

    "course access": [
        "After successful enrollment, your course should appear in your student dashboard."
    ],

    "refund": [
        "Refund eligibility depends on the course refund policy. Please contact customer support for assistance."
    ],

    "refund policy": [
        "Please check the refund policy of the selected course before requesting a refund."
    ],

    "refund status": [
        "You can contact customer support to check the status of your refund."
    ],

    "cancel course": [
        "Course cancellation depends on the course policy. Please contact customer support for assistance."
    ],

    "cancel enrollment": [
        "Please contact customer support if you want to cancel your enrollment."
    ],

    "customer support": [
        "Our customer support team can help with courses, enrollment, payments, technical problems and certificates."
    ],

    "contact support": [
        "Please use the Contact Us section to reach our customer support team."
    ],

    "contact": [
        "You can contact our support team through the Contact Us section."
    ],

    "email": [
        "Please check the Contact Us section for the official support email address."
    ],

    "phone": [
        "Please check the Contact Us section for the latest support phone number."
    ],

    "working hours": [
        "Customer support working hours are available in the Contact Us section."
    ],

    "feedback": [
        "We appreciate your feedback. Please use the Feedback section to share your experience."
    ],

    "complaint": [
        "We are sorry for the inconvenience. Please contact customer support with details of your problem."
    ],

    "suggestion": [
        "Thank you for your suggestion. We appreciate your help in improving our learning platform."
    ],

    "thank": [
        "You're welcome!",
        "Happy to help!"
    ],

    "thanks": [
        "You're welcome! Have a great day."
    ],

    "bye": [
        "Goodbye! Have a great day and happy learning!",
        "Thank you for using our online learning platform."
    ],

    "goodbye": [
        "Goodbye! Feel free to contact us again if you need help."
    ]
}





print("Loading Transformer AI model...")
print("First run may take some time.")


try:

    ai_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    transformer_available = True

    print("Transformer model loaded successfully.")

except Exception as error:

    ai_model = None

    transformer_available = False

    print("Transformer model could not be loaded.")
    print("Error:", error)




def find_faq_response(message):

    message_lower = message.lower().strip()

    # First check exact multi-word phrases
    sorted_keywords = sorted(
        faq.keys(),
        key=len,
        reverse=True
    )

    for keyword in sorted_keywords:

        if keyword in message_lower:

            return random.choice(faq[keyword])

    # NLTK token matching
    tokens = preprocess_text(message)

    for keyword in sorted_keywords:

        keyword_tokens = preprocess_text(keyword)

        if not keyword_tokens:
            continue

        matched_words = 0

        for word in keyword_tokens:

            if word in tokens:
                matched_words += 1

        if matched_words == len(keyword_tokens):

            return random.choice(faq[keyword])

    return None



# TRANSFORMER RESPONSE


def generate_ai_response(message):

    if not transformer_available:

        return (
            "I'm sorry, I could not understand your question. "
            "Please ask about courses, enrollment, fees, "
            "classes, assignments, exams or certificates."
        )

    # Build conversation context
    context = ""

    if conversation_history:

        recent_history = conversation_history[-4:]

        context = "\n".join(
            [
                f"User: {item['user']}\nBot: {item['bot']}"
                for item in recent_history
            ]
        )

    prompt = f"""
You are an online course customer support chatbot.

Answer the user's question clearly and briefly.

You can help with:
courses, enrollment, fees, payments, classes,
study materials, assignments, exams, certificates,
account problems and technical support.

Conversation history:
{context}

Current user question:
{message}

Give a helpful customer-support answer.
Do not invent specific prices, dates, phone numbers or email addresses.
If the question needs account-specific information, tell the user to contact customer support.

Answer:
"""

    try:

        result = ai_model(
            prompt,
            max_new_tokens=80,
            do_sample=False
        )

        response = result[0]["generated_text"].strip()

        if not response:

            return "Please contact customer support for more information."

        return response

    except Exception as error:

        print("Transformer error:", error)

        return (
            "Sorry, I could not process your question right now. "
            "Please try again."
        )



# CHATBOT RESPONSE


def chatbot_response(message):

    global conversation_history

    # Empty message
    if not message or not message.strip():

        return "Please type a question."


    # NLTK preprocessing
    tokens = preprocess_text(message)

    print("User tokens:", tokens)


    # Check FAQ first
    faq_response = find_faq_response(message)

    if faq_response:

        bot_response = faq_response

    else:

        # Use Transformer when FAQ doesn't match
        bot_response = generate_ai_response(message)


    # Save conversation context
    conversation_history.append({
        "user": message,
        "bot": bot_response
    })


    # Keep only last 10 conversations
    if len(conversation_history) > 10:

        conversation_history = conversation_history[-10:]


    return bot_response


# HOME PAGE


@app.route("/")
def home():

    return render_template("index.html")



# CHAT API


@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "response": "Invalid request."
            }), 400


        user_message = data.get("message", "").strip()


        if not user_message:

            return jsonify({
                "response": "Please type a message."
            }), 400


        # Generate response
        bot_response = chatbot_response(user_message)


        # Save interaction in SQLite
        save_chat(
            user_message,
            bot_response
        )


        return jsonify({
            "response": bot_response
        })


    except Exception as error:

        print("Chat API error:", error)

        return jsonify({
            "response": "Sorry, something went wrong."
        }), 500


# CLEAR CONVERSATION


@app.route("/clear", methods=["POST"])
def clear_conversation():

    global conversation_history

    conversation_history = []

    return jsonify({
        "message": "Conversation cleared."
    })



# RUN APPLICATION


if __name__ == "__main__":

    # Create database
    create_database()


    print("AI-POWERED CHATBOT")

    print("NLP: NLTK")
    print("AI: Transformers")
    print("Framework: Flask")
    print("Database: SQLite")
   
    print("Open: http://127.0.0.1:5000")
   

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
