import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
lemmatizer = WordNetLemmatizer()

model = load_model('chatbotmodel.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Predict



def clean_message(message):
    message_words = nltk.word_tokenize(message)
    from training import ignoreLetters
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words if word not in ignoreLetters]
    return message_words

def bag_of_words(message):
    message_words = clean_message(message)
    bag = [0] * len(words)
    for w in message_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(message, model):
    bow = bag_of_words(message)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
    return result


print("GO CHATBOT IS READY")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)