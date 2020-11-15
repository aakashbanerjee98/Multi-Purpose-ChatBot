import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

try:
  
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
#Preprosessing our data so that we can build the model
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

#Since the tflearn takes only np arrays so they are converted
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

#Building the model with the help of tflearn
ann = tflearn.input_data(shape=[None, len(training[0])])#input layer
ann = tflearn.fully_connected(ann, 8)#1st hidden layer
ann = tflearn.fully_connected(ann, 8)#2nd Hidden layer
ann = tflearn.fully_connected(ann, len(output[0]), activation="softmax")#output layer
ann = tflearn.regression(ann)

model = tflearn.DNN(ann)

#Training our model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

#This function takes the input word and tokenizes and stems it for our model
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

#This is the main fuction that takes the user input and returns the suitable output to the user
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #Since the model outputs the probabilty of each respone we only select the ones that are higher than a particular thresshold
        if results[results_index] > 0.7:

          for tg in data["intents"]:
              if tg['tag'] == tag:
                  responses = tg['responses']
          print(random.choice(responses))
        else:
          print("I dont quite understand, try again")



chat()