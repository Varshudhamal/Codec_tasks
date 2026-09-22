from flask import Flask, render_template, request
from transformers import pipeline
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from datetime import datetime, timezone
import os


# FLASK APP


app = Flask(__name__)


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/"
)

DATABASE_NAME = os.getenv(
    "MONGO_DATABASE",
    "sentiment_analyzer"
)

COLLECTION_NAME = os.getenv(
    "MONGO_COLLECTION",
    "reviews"
)


# MONGODB CONNECTION


mongodb_connected = False
client = None
db = None
reviews_collection = None

try:

    print("Connecting to MongoDB...")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    # Test connection
    client.admin.command("ping")

    db = client[DATABASE_NAME]

    reviews_collection = db[COLLECTION_NAME]

    mongodb_connected = True

    print("MongoDB Connected Successfully!")
    print("Database:", DATABASE_NAME)
    print("Collection:", COLLECTION_NAME)

except PyMongoError as error:

    print("MongoDB Connection Error:")
    print(error)


# LOAD HUGGING FACE SENTIMENT MODE

print()
print("Loading Hugging Face sentiment model...")
print("First run may download the model.")
print()

sentiment_model = None
model_available = False

try:

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    model_available = True

    print("Hugging Face sentiment model loaded successfully!")

except Exception as error:

    print("Transformer model could not be loaded.")
    print("Error:", error)


# CONVERT SENTIMENT LABEL


def convert_sentiment(label):

    label = str(label).upper().strip()

    if label == "LABEL_0":
        return "Negative"

    if label == "LABEL_1":
        return "Neutral"

    if label == "LABEL_2":
        return "Positive"

    if "NEGATIVE" in label:
        return "Negative"

    if "NEUTRAL" in label:
        return "Neutral"

    if "POSITIVE" in label:
        return "Positive"

    return label.title()



# GET SENTIMENT STATISTICS


def get_statistics():

    statistics = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    if not mongodb_connected:
        return statistics

    try:

        statistics["Positive"] = reviews_collection.count_documents(
            {"sentiment": "Positive"}
        )

        statistics["Negative"] = reviews_collection.count_documents(
            {"sentiment": "Negative"}
        )

        statistics["Neutral"] = reviews_collection.count_documents(
            {"sentiment": "Neutral"}
        )

    except PyMongoError as error:

        print("Statistics Error:")
        print(error)

    return statistics



# GET RECENT REVIEWS


def get_recent_reviews():

    if not mongodb_connected:
        return []

    try:

        reviews = list(
            reviews_collection.find(
                {},
                {
                    "_id": 0,
                    "text": 1,
                    "sentiment": 1,
                    "score": 1,
                    "created_at": 1
                }
            )
            .sort(
                "created_at",
                -1
            )
            .limit(5)
        )

        return reviews

    except PyMongoError as error:

        print("Recent Reviews Error:")
        print(error)

        return []



# HOME PAGE


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = ""
    score = 0
    text = ""
    error = ""
    success = ""

    
    # POST REQUEST


    if request.method == "POST":

        text = request.form.get(
            "text",
            ""
        ).strip()

        
        # EMPTY TEXT
        

        if not text:

            error = (
                "Please enter a review or social media text."
            )

        # TEXT LENGTH
      

        elif len(text) > 5000:

            error = (
                "Text must not exceed 5000 characters."
            )

    
        # MODEL CHECK
        

        elif not model_available:

            error = (
                "AI sentiment model is not available. "
                "Please check your Transformers and PyTorch installation."
            )

        else:

            try:

             
                # SENTIMENT PREDICTION
               

                result = sentiment_model(
                    text,
                    truncation=True,
                    max_length=512
                )[0]

                raw_label = result["label"]

                confidence = float(
                    result["score"]
                )

                sentiment = convert_sentiment(
                    raw_label
                )

                score = round(
                    confidence * 100,
                    2
                )

               
                # TERMINAL OUTPUT
               

                print()
                print("------------------------------------------")
                print("Text:", text)
                print("Raw Label:", raw_label)
                print("Sentiment:", sentiment)
                print("Confidence:", score, "%")
                print("------------------------------------------")

                
                # SAVE TO MONGODB
                

                if mongodb_connected:

                    try:

                        reviews_collection.insert_one(
                            {
                                "text": text,
                                "sentiment": sentiment,
                                "score": score,
                                "created_at": datetime.now(
                                    timezone.utc
                                )
                            }
                        )

                        success = (
                            "Sentiment analyzed and saved "
                            "to MongoDB successfully."
                        )

                    except PyMongoError as mongo_error:

                        print(
                            "MongoDB Save Error:",
                            mongo_error
                        )

                        error = (
                            "Sentiment analyzed, but the "
                            "result could not be saved to MongoDB."
                        )

                else:

                    error = (
                        "Sentiment analyzed, but MongoDB "
                        "is not connected."
                    )

            except Exception as analysis_error:

                print(
                    "Analysis Error:",
                    analysis_error
                )

                error = (
                    "Unable to analyze the text."
                )

  
    # GET STATISTICS
   

    statistics = get_statistics()

    positive_count = statistics["Positive"]
    negative_count = statistics["Negative"]
    neutral_count = statistics["Neutral"]

    total_count = (
        positive_count
        + negative_count
        + neutral_count
    )

    
    # GET RECENT REVIEWS


    recent_reviews = get_recent_reviews()

   
    # SEND DATA TO HTML


    return render_template(
        "index.html",

        sentiment=sentiment,

        score=score,

        text=text,

        error=error,

        success=success,

        positive_count=positive_count,

        negative_count=negative_count,

        neutral_count=neutral_count,

        total_count=total_count,

        mongodb_connected=mongodb_connected,

        model_available=model_available,

        recent_reviews=recent_reviews
    )



# RUN FLASK APPLICATION


if __name__ == "__main__":

    print()
    print("***************************************")
    print("       AI-BASED SENTIMENT ANALYZER")
    print("***************************************")
    print("AI: Hugging Face Transformers")
    print("Model: CardiffNLP Twitter RoBERTa")
    print("Framework: Flask")
    print("Database: MongoDB")
    print("Output: Positive / Negative / Neutral")
    print("Report: Sentiment Distribution")
    print("**************************************")
    print("Open: http://127.0.0.1:5000")
    print("**************************************")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )