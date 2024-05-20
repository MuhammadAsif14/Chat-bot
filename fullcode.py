import tensorflow as tf
import keras
import nltk
import pickle
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

print("words: \n\n")
print(words)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

"""
print(documents)
print("\n\n")
print(classes)
print("\n\n")
print(training)
"""
random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]#features or words in vocab
trainY = training[:, len(words):]#labels or classes

"""
print("trainX:  ")
print(trainX)

print("trainy:  ")
print(trainY)
print("\n\n")
print(len(trainY[0]))
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history=model.fit(trainX, trainY, epochs=300, batch_size=64, verbose=1)

train_loss = history.history['loss']
plt.plot(train_loss, label='Training Loss')
# If validation data was used, plot it as well.
# plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.save('chatbotmodel.h5',history)
model.summary()
model.layers

print('Done')


kf = KFold(n_splits=5, shuffle=True)

# Track model accuracy across folds
acc_scores = []

# Perform cross-validation
for train_index, test_index in kf.split(trainX):
    X_train, X_test = trainX[train_index], trainX[test_index]
    y_train, y_test = trainY[train_index], trainY[test_index]

    # Train model on each fold
    model.fit(X_train, y_train)

    # Evaluate model on test fold and store accuracy
    _, acc = model.evaluate(X_test, y_test)
    acc_scores.append(acc)
# Convert scores to a NumPy array
scores_array = np.array(acc_scores)

# Calculate mean and standard deviation
mean_accuracy = np.mean(scores_array)
std_accuracy = np.std(scores_array)
print(f"Accuracies:{scores_array}")
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Accuracy: {std_accuracy}")


kf = KFold(n_splits=10, shuffle=True)

# Track model accuracy across folds
acc_scores = []

# Perform cross-validation
for train_index, test_index in kf.split(trainX):
    X_train, X_test = trainX[train_index], trainX[test_index]
    y_train, y_test = trainY[train_index], trainY[test_index]

    # Train model on each fold
    model.fit(X_train, y_train)

    # Evaluate model on test fold and store accuracy
    _, acc = model.evaluate(X_test, y_test)
    acc_scores.append(acc)
# Convert scores to a NumPy array
scores_array = np.array(acc_scores)

# Calculate mean and standard deviation
mean_accuracy = np.mean(scores_array)
std_accuracy = np.std(scores_array)
print(f"Accuracies:{scores_array}")
print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation of Accuracy: {std_accuracy}")



from keras.models import load_model

model = load_model('chatbotmodel.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Predict
def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for result in results:
        return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
    return return_list


def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
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
    message = input("YOU: ")
    ints = predict_class(message,model)
    res = get_response(ints, intents)
    print("BOT: "+res)