import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)
    new_words = []
    detok = TreebankWordDetokenizer()

    # probabilities of applying transformations
    synonym_prob = 0.20
    typo_prob = 0.08

    keyboard_neighbors = {
        'a': 'qs',
        's': 'adw',
        'd': 'sfe',
        'e': 'drw',
        'r': 'etf',
        't': 'ryg',
        'n': 'bhm',
        'm': 'jkn',
        'i': 'uok',
        'o': 'ipk',
    }

    for word in words:
        w = word

        if random.random() < synonym_prob:
            synsets = wordnet.synsets(w)
            if synsets:
                lemmas = synsets[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name()
                    if synonym != w and synonym.isalpha():
                        w = synonym

        if random.random() < typo_prob:
            chars = list(w)
            idxs = list(range(len(chars)))
            random.shuffle(idxs)
            replaced = False

            for idx in idxs:
                ch = chars[idx]
                if ch.lower() in keyboard_neighbors:
                    chars[idx] = random.choice(keyboard_neighbors[ch.lower()])
                    replaced = True
                    break

            if replaced:
                w = "".join(chars)

        new_words.append(w)

    example["text"] = detok.detokenize(new_words)


    ##### YOUR CODE ENDS HERE ######

    return example
