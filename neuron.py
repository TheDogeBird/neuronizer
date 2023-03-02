import random
import threading
import tensorflow as tf
from nltk.corpus import wordnet
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Neuron(tf.keras.layers.Layer):
    def __init__(self):
        super(Neuron, self).__init__()
        self.words = set()
        self.count = 0
        self.dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # Skip short words
        if len(inputs) <= 2:
            print(f"Skipping short word '{inputs}'")
            return

        # Get a random definition for the given word
        synsets = wordnet.synsets(inputs)
        if not synsets:
            print(f"No synsets found for word '{inputs}'")
            return
        definition = wordnet.synset(random.choice(synsets).name()).definition()

        # Print the definition and activate all associated neurons
        print(f"{inputs}: {definition}")
        output = self.dense_layer(tf.constant([[1.0]]))
        return output



class WordNetNeuron(Neuron):
    def __init__(self):
        super(WordNetNeuron, self).__init__()
        self.associated_neurons = []

    def add_associated_neuron(self, neuron):
        self.associated_neurons.append(neuron)


class NLPNeuron(Neuron):
    def __init__(self):
        super(NLPNeuron, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=16)

    def call(self, inputs, training=None, mask=None):
        # Skip short words
        if len(inputs) <= 2:
            print(f"Skipping short word '{inputs}'")
            return

        # Get word embedding
        embedding = self.embedding(tf.constant([inputs]))

        # Print the embedding vector and activate all associated neurons
        print(f"{inputs}: {embedding.numpy()}")
        output = self.dense_layer(embedding)
        return output


def input_loop(neuron):
    while True:
        input_text = input("Enter some text: ")
        words = nltk.word_tokenize(input_text)
        for word in words:
            neuron.call(word)


if __name__ == "__main__":
    # Create a WordNetNeuron instance and add some associated neurons
    neuron1 = WordNetNeuron()
    neuron2 = WordNetNeuron()
    neuron1.add_associated_neuron(neuron2)
    neuron2.add_associated_neuron(neuron1)

    # Create a WordNetNeuron instance and add some associated neurons
    neuron3 = WordNetNeuron()
    neuron4 = WordNetNeuron()
    neuron3.add_associated_neuron(neuron4)
    neuron4.add_associated_neuron(neuron3)

    # Start input loop threads for each neuron
    thread1 = threading.Thread(target=input_loop, args=(neuron1,))
    thread2 = threading.Thread(target=input_loop, args=(neuron3,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
