import nltk
import pickle
import numpy as np
import re
import json
from nltk.stem.porter import PorterStemmer 



with open('src/pickle/label_encoder.pkl', 'rb') as file:
    lb = pickle.load(file)

with open('src/pickle/logistic_regresion.pkl', 'rb') as file:
    lg = pickle.load(file)

with open('src/pickle/vector.pkl', 'rb') as file:
    vector = pickle.load(file)


stopwords = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


all_metrics = []
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = vector.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(lg.predict(input_vectorized))

    all_metrics.append({
        'sentence': sentence,
        'emotion': predicted_emotion,
        'label': int(label),
    })



sentences = [
            "i didnt feel humiliated",
            "i feel strong and good overall",
            "im grabbing a minute to post i feel greedy wrong",
            "He was speechles when he found out he was accepted to this new job",
            "This is outrageous, how can you talk like that?",
            "I feel like im all alone in this world",
            "He is really sweet and caring",
            "You made me very crazy",
            "i am ever feeling nostalgic about the fireplace i will know that it is still on the property",
            "i am feeling grouchy",
            "He hates you"
            ]
for sentence in sentences:
    predict_emotion(sentence)

with open('metrics.json', 'w') as file:
    json.dump(all_metrics, file, indent=4)