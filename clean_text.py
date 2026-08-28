import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

if __name__ == "__main__":
    sample = "Dear Support Team, I hope this message reaches you. My payment failed!"
    print(clean_text(sample))