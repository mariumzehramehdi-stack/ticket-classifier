# SupportIQ

### AI-Powered Customer Support Ticket Classification

SupportIQ is an NLP-based machine learning application that automatically classifies customer support tickets by **category** and **priority**.

It combines a trained machine learning pipeline with a **FastAPI REST API** and a web-based frontend to provide real-time predictions and confidence scores.

---

## Overview

Support teams receive large numbers of customer requests that need to be categorized and prioritized before they can be handled efficiently.

SupportIQ automates this initial routing process by analyzing the text of a support ticket and predicting:

* **Ticket Category** — one of 10 categories
* **Ticket Priority** — one of 3 priority levels
* **Prediction Confidence** — confidence score for each prediction

---

## How It Works

```text
Customer Support Ticket
          │
          ▼
     Text Cleaning
          │
          ▼
   TF-IDF Vectorization
          │
          ▼
   ┌───────────────────┐
   │                   │
   ▼                   ▼
Category Model     Priority Model
   │                   │
   ▼                   ▼
Category            Priority
Confidence          Confidence
```

The frontend sends the customer's message to the FastAPI backend.

The backend:

1. Cleans the input text.
2. Converts the text into TF-IDF features.
3. Passes the features to the trained category and priority models.
4. Returns the predictions and confidence scores to the frontend.

---

## Features

* Automated support ticket classification
* 10 ticket categories
* 3 priority levels
* NLTK-based text preprocessing
* TF-IDF vectorization with 30,000 features
* Category and priority confidence scores
* FastAPI REST API
* Interactive web interface
* Responsive frontend
* Docker containerization

---

## Machine Learning

The project uses traditional NLP techniques for text classification.

### Text Processing

Customer messages are cleaned and prepared before being converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The final vectorizer contains approximately **30,000 features**.

### Classification

Two separate supervised learning models are used:

| Model          | Purpose                              |
| -------------- | ------------------------------------ |
| Category Model | Predicts the support ticket category |
| Priority Model | Predicts the ticket priority         |

The trained models and vectorizer are saved using Joblib and loaded by the FastAPI application.

---

## Model Performance

The final models were evaluated on a held-out test set.

| Classification Task | Accuracy |
| ------------------- | -------: |
| Ticket Category     |  **53%** |
| Ticket Priority     |  **65%** |

Additional evaluation metrics include:

* Precision
* Recall
* F1-score
* Accuracy

The category task is more challenging because it involves more classes and some categories contain overlapping language.

---

## Dataset

 The project uses a customer support ticket dataset containing approximately 28,000+ tickets.

The data was filtered to English-language tickets during preprocessing before model training. The dataset contains customer support messages together with labels used to train the category and priority classifiers.

---

## Tech Stack

### Machine Learning

* Python
* Scikit-learn
* NLTK
* Pandas
* NumPy
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* HTML
* CSS
* JavaScript

### Development & Deployment

* Git
* GitHub
* Docker
* Python Virtual Environment

---

## Project Structure

```text
ticket-classifier/
│
├── clean_text.py          # Text preprocessing
├── main.py                # FastAPI application
├── index.html             # Frontend interface
│
├── queue_model.pkl        # Trained category model
├── priority_model.pkl     # Trained priority model
├── vectorizer.pkl         # Trained TF-IDF vectorizer
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git exclusions
└── README.md              # Project documentation
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ticket-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### 6. Open the frontend

Open `index.html` in your browser and enter a customer support message.

---

## Run with Docker

Docker packages the FastAPI application and its dependencies into a container so the application can run in a consistent environment.

### Build the image

```bash
docker build -t supportiq .
```

### Run the container

```bash
docker run -p 8000:8000 supportiq
```

The API will then be available at:

```text
http://localhost:8000
```

---

## Example

### Input

```text
My payment was charged twice and I need help getting a refund.
```

### Output

```text
Predicted Category: ...
Category Confidence: ...%

Predicted Priority: ...
Priority Confidence: ...%
```

The actual predictions depend on the trained models and the input ticket.

---

## Current Scope

The current version focuses on text-based ticket classification using traditional NLP and machine learning techniques.

The application demonstrates an end-to-end workflow from **customer input to machine learning prediction through a REST API**.

---

## Future Improvements

* Experiment with transformer-based NLP models.
* Improve classification performance through additional data and hyperparameter tuning.
* Add ticket history and analytics.
* Add database integration.
* Add authentication and role-based access.
* Deploy the application to a production environment.

---

## Author

**Marium Zehra**

BS Computer Science Student
 
          

