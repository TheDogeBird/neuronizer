import random
import threading
from nltk.corpus import wordnet
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class Neuron:
    def __init__(self):
        self.words = set()
        self.count = 0
        self.associated_neurons = set()

    def activate(self, word):
        # Skip short words
        if len(word) <= 2:
            print(f"Skipping short word '{word}'")
            return

        # Get a random definition for the given word
        synsets = wordnet.synsets(word)
        if not synsets:
            print(f"No synsets found for word '{word}'")
            return
        definition = wordnet.synset(random.choice(synsets).name()).definition()

        # Print the definition and activate all associated neurons
        print(f"{word}: {definition}")
        for neuron in self.associated_neurons:
            neuron.activate(word)

def input_loop(neuron):
    while True:
        input_text = input("Enter some text: ")
        words = nltk.word_tokenize(input_text)
        for word in words:
            neuron.activate(word)


if __name__ == "__main__":
    neuron = Neuron()
    input_thread = threading.Thread(target=input_loop, args=(neuron,))
    input_thread.start()
    input_thread.join()
