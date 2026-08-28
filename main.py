#tools that ur api need  
from fastapi import FastAPI # with this ur frontend can send "my payment failed" to api and then it will send back the prediction by ml model 
from fastapi.middleware.cors import CORSMiddleware#without this if you later open yur webpage and call api browser will block random request CORS middleware tells your API:

"///It's okay for this webpage to send requests to me."
from pydantic import BaseModel#In the Pydantic library, BaseModel is the core class used to define schemas, validate data structures, and manage serialization in Python.
import joblib#loads ur trained model

from clean_text import clean_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = joblib.load("vectorizer.pkl")#calls already  trained model files from diak happend only once when servr starts
queue_model = joblib.load("queue_model.pkl")#
priority_model = joblib.load("priority_model.pkl")#

class TicketRequest(BaseModel):
    text: str#only str data will be take in other data type will be rejected

@app.post("/classify")  #this creates An API endpoint is a specific URL    where an API receives requests and sends responses back to a client application.
 
def classify_ticket(request: TicketRequest):
   
    cleaned = clean_text(request.text)
    vectorized = vectorizer.transform([cleaned])

    queue_prediction = queue_model.predict(vectorized)[0]
    priority_prediction = priority_model.predict(vectorized)[0]

    queue_probs = queue_model.predict_proba(vectorized)[0]
    priority_probs = priority_model.predict_proba(vectorized)[0]

    queue_confidence = round(float(max(queue_probs)) * 100, 1)
    priority_confidence = round(float(max(priority_probs)) * 100, 1)

    return {
        "queue": str(queue_prediction),
        "queue_confidence": queue_confidence,
        "priority": str(priority_prediction),
        "priority_confidence": priority_confidence,
        "needs_review": bool(queue_confidence < 50),
    }
@app.get("/")
def root():
    return {"status": "Ticket classifier API is running"}