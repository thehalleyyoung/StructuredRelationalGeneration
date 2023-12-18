
import torch
import torch.nn as nn
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
import os
import itertools
from transformers import BertTokenizer, BertForMaskedLM
import pickle
import random
import numpy as np
from inspect import getmembers, isfunction
import itertools
import poetryrelations
from torch.distributions import Categorical
from genfsa2 import *
import pronouncing
from importlib import reload
import pdfkit
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import copy
from nltk.corpus import stopwords
from poemstrucvae import Predictor
from gensim.models.wrappers import FastText

# Initialize a Predictor instance with specific parameters
predictor = Predictor(19, 1000, 10, 9)
# Load the trained state dictionary into the predictor
predictor.load_state_dict(torch.load("models/predict_program.pth"))

# Dictionary to store similarity information
are_similar = {}

# Retrieve all functions from the poetryrelations module except 'word'
functions_list = [o for o in getmembers(poetryrelations) if isfunction(o[1]) if o[0] != "word"]
# Extract the names of the functions from the functions_list
functions_name_list = [o[0] for o in functions_list if o[0] != "word"]

# Function to check if tokens are simple (not subwords)
def simpleToks(poem):
    for i in range(len(poem)):
        if poem[i].startswith("##"):
            return False
    return True

# Load BERT tokenizer and model if not already in globals
if "berttok" not in globals():
    berttok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # Tokenize and filter poems based on syllable count
    poems = [(berttok.tokenize(i.strip("\n"))) for i in open("PoetryFoundationData.csv", "r+").readlines() if sum([len(hyphenate.hyphenate_word(x)) for x in (i.split(" "))]) in range(6,14)]
    poems = [i for i in poems if len(i) >= 4]
    # Load preprocessed rhyme and meter sets
    (poems_rhyme_sets, poems_rhyme_meter_sets, poems_meter_sets) = pickle.load(open("vocab/poemssets1.pcl", "rb"))

    # Filter human-readable poems
    poems_human = [i for i in poems if simpleToks(i) and len([k for k in i if k.isalpha()]) >= 4]

    # Load additional preprocessed data
    (vocab_sylls_ids, rhyme_word_set, rhyme_meter_word_set) = pickle.load(open("vocab/poemssets2.pcl", "rb"))
    (lastWords, all_words) = pickle.load(open("vocab/all_words.pcl", "rb"))
    # Load FastText model for word similarity
    modelsim = FastText.load_fasttext_format('cc.en.300.bin')

    # Re-import BERT model, which may be unnecessary
    from transformers import BertTokenizer, BertForMaskedLM
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load list of swear words
swear_words = [i.strip("\n") for i in open("swear.txt").readlines()]
# Convert swear words to BERT token ids
swear_words = berttok.convert_tokens_to_ids([k for k in swear_words if k in berttok.vocab])

# Concatenate lists of lists into a single list
def concat(xss):
    new = []
    for xs in xss:
        new.extend(xs)
    return new

# Create a reverse dictionary for BERT vocabulary
inverseVocab = {v:k for k,v in list(berttok.vocab.items()) if k.isalpha()}

# Define rhyme pattern and meter for the poem
rhyme_pattern = [0,0,1,1,2,2,3,3,0,0]
meter = [10,10,8,8,10,10,8,8,10,10]
# Initialize syllable ids for vocabulary
vocab_sylls_ids = {i:[] for i in range(0, 15)}
for i in list(berttok.vocab.keys()):
    if not i.isalpha():
        vocab_sylls_ids[berttok.vocab[i]] = 0
    else:
        vocab_sylls_ids[berttok.vocab[i]] = len(hyphenate.hyphenate_word(i))

# Function to sample n elements from a list
def sample(xs, n):
    return random.sample(xs, min(n, len(xs)))

# Load most similar words data
most_similar = pickle.load(open("vocab/most_similar.pcl", "rb"))

# Function to calculate the score of a sentence
def score(sentence):
    sentence2 = gpt2tok.convert_tokens_to_ids(berttok.convert_ids_to_tokens(sentence))
    tensor_input = torch.tensor(sentence).view(1,-1)
    tokenize_input = berttok.convert_ids_to_tokens(sentence)
    sentence_loss=0.
    loss = model(tensor_input, labels=tensor_input)[0]
    return math.exp(loss)

# Function to get the best line based on maximum probabilities
def maxProbabilities(lines_id,  possibilities):
    a = max(possibilities, key = lambda i: score(lines_id + berttok.convert_tokens_to_ids(i)))
    return a

# Convert English stopwords to BERT token ids
stop = berttok.convert_tokens_to_ids([q for q in stopwords.words("english") if q in list(berttok.vocab.keys())])
# Identify bad tokens in the BERT vocabulary
bad = [v for (k,v) in list(berttok.vocab.items()) if any([q in k for q in "â€œâ€"])]
bad_chars = "â€œâ€"


# Function to replace a word with the most probable alternative
def replaceWordWithMostProbableAlternative(copy_best_line, lines_id, best_line, j, prev_len):
    # Convert lines_id to a PyTorch tensor for model input
    tokens_tensor = torch.Tensor(lines_id).view(1,-1).long()

    # Get the length of lines_id
    c = len(lines_id)

    # Extract the word to be replaced from the copy of the best line
    word = copy_best_line[j]

    # Determine the word length based on hyphenation; 0 if not an alphabetical word
    word_len = 0 if not word.isalpha() else len(hyphenate.hyphenate_word(word))

    # Print the word details for debugging
    print((word_len, j, word, best_line))

    # Predict all tokens with the model
    with torch.no_grad():
        # Mask the current word in the tensor
        tokens_tensor[0, prev_len + j] = berttok.vocab["[MASK]"]

        # Run the model to get predictions
        outputs = model(tokens_tensor, labels=tokens_tensor)
        loss, predictions = outputs[:2]

        # Focus on the predictions for the masked word
        predictions = predictions[0, prev_len + j, :]

        # Adjust the predictions based on various criteria
        predictions *= 20
                for (k,v) in vocab_sylls_ids.items():
            if word in most_similar and  k in most_similar[word] and not (word in stopwords.words("english")):
                predictions[k] -= 10
            if k in stop and not lines_id[prev_len + j] in stop:
                predictions[k] -= 100
            if k in swear_words:
                predictions[k] = -1e20
            if v != word_len:
                predictions[k] = -1e20
            if k == lines_id[prev_len + j - 1] or k == lines_id[prev_len + j - 2] or k == lines_id[prev_len + j - 3]:
                predictions[k] -= 100
            if k in bad:
                predictions[k] = -1e20
        predictions[berttok.vocab["x"]] = -1e20

        # Choose the most probable alternative
        predicted_index = Categorical(logits=predictions).sample()
        predicted_token = berttok.convert_ids_to_tokens([predicted_index.item()])[0]

        # Replace the word in the best line
        best_line[j] = predicted_token

        # Ensure the new word is different from the original if it's not a stop word
        if not (word_len == 0 or word in stopwords.words("english")):
            assert(best_line[j] != word)
        
        # Update the lines_id with the new word's index
        lines_id[prev_len + j] = (predicted_index.item())


# Function to return the last word of a line
def lastWord(j):
    return [i for i in j if i.isalpha()][-1]

# Function to get the total size of a dictionary entry
def totSize(dic, word):
    if word not in dic:
        return 0
    else:
        return len(dic[word])

# Function to format text with a specific color
def color_format(color):
    return """
      <p style="color: """ + color + """; font-family: 'Liberation Sans',sans-serif">
        {}
      </p>
    """

# Function to check if a string is 'good' based on certain conditions
def good(xs):
    return "'" not in xs and len([k for k in xs if k.isalpha()]) >= max(len(xs) - 2,3)

# Function to check if a word rhymes with any of the last words
def prevRhyme(i, lastwords):
    return any([lastWord(i) in pronouncing.rhymes(lastword) for lastword in lastwords])

# Filter poems based on meter sets
for i in range(2,20):
    print(len(poems_meter_sets[i]))
    poems_meter_sets[i] = [k for k in poems_meter_sets[i] if lastWord(k) in rhyme_meter_word_set[i]]
    print(len(poems_meter_sets[i]))

# Set a starting index for generated poems
z = 5300
# Initialize a list to store time records
times = []

# Function to check for previous patterns in the rhyme meter word set
def hasPrevPat(rhyme_meter_word_set, prev_pat):
    return [k[prev_pat] for k in rhyme_meter_word_set.values() if prev_pat in k]

# Function to generate a poem with a given index
def genPoem(z):
    # Code for generating a poem
    def genPoem(z):
    latent = torch.randn(50)
    (spacing_out, rels_out) = predictor.decoder(latent)
    rhyme_pattern = []
    meter = []
    spacings = []
    for k in range(10):
        spacing = int(torch.argmax(spacing_out.view(10,10)[:,k])))
        if spacing == 0 and len([i for i in spacings[-3:] if i == 0]) == 3:
            spacing = 1
        if rels_out.view(10,9)[k,functions_name_list.index("endrhyme")] > 0.3:
            if spacing == 0:
                rhyme_pattern.append(max(rhyme_pattern) + 1 if len(rhyme_pattern) > 0 else 0)
            else:
                rhyme_pattern.append(rhyme_pattern[-1*spacing])
        else:
            rhyme_pattern.append(max(rhyme_pattern) + 1 if len(rhyme_pattern) > 0 else 0)
        if rels_out.view(10,9)[k,functions_name_list.index("possSameMeter")] > 0.4:
            if spacing == 0:
                meter.append(random.choice([8,10,12]))
            else:
                meter.append(rhyme_pattern[-1*spacing])
        else:
            meter.append(random.choice([8,10,12]))
        spacings.append(spacing)

    print("rhyme_pattern " + str(rhyme_pattern))

    start_time = time.time()
    max_score = -1e20
    lines = []
    lines_id = []
    lastwords = []
    lines_id_cur = []
    for i in range(10):
        totScore = 1100
        line_score = 1e8
        full_score = 1e9
        index = 0
        # Loop until you find a line satisfying the desired formal characteristics and with low perplexity
        while (line_score > 4.5 or full_score > 3) and index < 1000:
            index += 1
            print("in again")
            print("Q_ is " + str(i))
            if rhyme_pattern[i] != None and meter[i] != None and rhyme_pattern[i] in rhyme_pattern[:i]:
                prev_pat = [lastWord(lines[k]) for k in range(i) if rhyme_pattern[k] == rhyme_pattern[i]][-1]
                try:
                    possibilities = sample(list(filter(lambda j: good(j) and not (lastWord(j) in lastwords), concat(rhyme_meter_word_set[meter[i]][prev_pat]))), 5)
                    if possibilities == []:
                        possibilities = sample(list(filter(lambda j: good(j) and not lastWord(j), concat(concat(hasPrevPat(rhyme_meter_word_set, prev_pat))))), 5)
                    if possibilities == []:
                        possibilities = sample(list(filter(lambda j: not prevRhyme(j,lastwords), sorted([k for k in poems_meter_sets[meter[i]]], key = lambda j: totSize(rhyme_word_set, lastWord(j)), reverse = True)))[:2000], 5)
                except:
                    print("in except")
                    possibilities = sample(list(filter(lambda j: good(j) and not (lastWord(j) in lastwords), concat(concat(hasPrevPat(rhyme_meter_word_set, prev_pat))))), 5)
                    if possibilities == []:
                        possibilities = sample(list(filter(lambda j: not prevRhyme(j,lastwords), sorted([k for k in poems_meter_sets[meter[i]]], key = lambda j: totSize(rhyme_word_set, lastWord(j)), reverse = True)))[:2000], 5)
                best_line = maxProbabilities(lines_id, possibilities)
                print(possibilities)
            elif meter[i] != None:
                possibilities = sample(list(filter(lambda j: not prevRhyme(j,lastwords), sorted([k for k in poems_meter_sets[meter[i]]], key = lambda j: totSize(rhyme_word_set, lastWord(j)), reverse = True)))[:2000], 5)
                best_line = maxProbabilities(lines_id, possibilities)
            elif rhyme_pattern[i] != None  and rhyme_pattern[i] in rhyme_pattern[:i]:
                prev_pat = [k for k in range(i) if rhyme_pattern[k] == rhyme_pattern[i]][-1]
                possibilities = sample(list(filter(good, sorted([k for k in poems_rhyme_sets[prev_pat]], key = lambda j: totSize(meter_word_set, lastWord(j)), reverse = True)))[:4000], 5)
                best_line = maxProbabilities(lines_id, possibilities)
            else:
                possibilities = sample(list(filter(good, sorted([k for k in all_lines], key = lambda j: totSize(rhyme_word_set, lastWord(j)), reverse = True)))[:2000], 20)
            best_line_copy = copy.copy(best_line)
            best_line_alphas = [i for i in range(len(best_line)) if best_line[i].isalpha()][:-1] + [i for i in range(len(best_line)) if best_line[i].isalpha()][:-1]
            random.shuffle(best_line_alphas)
            prev_len = len(lines_id_cur)
            new_lines_id_cur = lines_id_cur + berttok.convert_tokens_to_ids(best_line)
            for j in best_line_alphas:
                replaceWordWithMostProbableAlternative(best_line_copy, new_lines_id_cur, best_line, j, prev_len)
            line_score = score([berttok.vocab[k] for k in best_line])

            full_score = score(new_lines_id_cur)
            print("fullscore " + str(full_score) + " linescore " + str(line_score))
        lines.append(best_line)
        lastwords.append(lastWord(best_line))
        lines_id_cur += berttok.convert_tokens_to_ids(best_line)
        print("Lines: " + str(lines))


    rhyme_pat = [0]
    for i in range(1,len(lines)):
        for j in range(i):
            if lastWord(lines[i]) in pronouncing.rhymes(lastWord(lines[j])):
                rhyme_pat.append(rhyme_pat[j])
                break
        if len(rhyme_pat) <= i:
            rhyme_pat.append(max(rhyme_pat) + 1)
        print(lines[:i + 1], rhyme_pat)
    string = ""

    # Colors represent structure
    lines = [" ".join([k for k in x if not any([z in k for z in bad_chars])]) for x in lines]
    colors = ["red", "green", "blue", "brown", "pink", "purple", "yellow", "black", "orange", "violet"]

    for (q, line) in enumerate(lines):
        print(q)
        string += color_format(colors[rhyme_pat[q]]).format(line)
    # Generate HTML to generate PDF
    open("generated-poems/poemsampled" + str(z) + ".html", "w+").write(string)
    # Make colored PDF
    pdfkit.from_url("generated-poems/poemsampled" + str(z) + ".html", "poems/poemsampled" + str(z) + ".pdf") 
    times.append(time.time() - start_time)


# Loop to generate multiple poems
z = 0
for q in range(z,z+1000):
    try:
        genPoem(q)
    except:
        print("error")
        time.sleep(1)
