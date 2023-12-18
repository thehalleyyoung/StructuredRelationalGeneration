
# Import necessary libraries
import spacy
from Phyme import Phyme as phyme
import hyphenate
import pronouncing

# Helper lambda function to clean and lower-case words
word = lambda i: "".join([k for k in i if k.isalpha()]).lower()

# Load the spaCy English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")

# Initialize Phyme library for rhyming
ph = phyme()

# Function to determine if two lines end with slant rhymes
def endSlantRhyme(x, x1_ind, x2_ind):
    # Extract the last words from the lines at the given indices
    word1 = "".join([i for i in x[x1_ind].split()[-1] if i.isalpha()])
    word2 = "".join([i for i in x[x2_ind].split()[-1] if i.isalpha()])
    
    # List of rhyme type functions to check
    rhyme_types = [ph.get_family_rhymes, ph.get_partner_rhymes, ph.get_perfect_rhymes, ph.get_substitution_rhymes]
    
    # Check if the last words of both lines rhyme using the rhyme type functions
    try:
        for rhyme_type in rhyme_types:
            if any([word1 in k for k in rhyme_type(word2).values()]):
                return True
    except:
        return False
    return False

# Function to determine if two lines end with perfect rhymes
def endrhyme(x, x1_ind, x2_ind):
    # Extract the last words from the lines at the given indices
    word1 = "".join([i for i in x[x1_ind].split()[-1] if i.isalpha()])
    word2 = "".join([i for i in x[x2_ind].split()[-1] if i.isalpha()])
    
    # Check if the last words of both lines rhyme using pronouncing library
    return word1 in pronouncing.rhymes(word2)

# Function to determine if two lines start with the same word
def sameStartWord(x, x1_ind, x2_ind):
    return word(x[x1_ind].split(" ")[0].lower()) in x[x2_ind].lower() or word(x[x2_ind].split(" ")[0].lower()) in x[x1_ind].lower()

# Function to determine if two lines contain the same non-stop word
def containsSameNonStopWord(x, x1_ind, x2_ind):
    # Tokenize and filter out stop words and punctuation from both lines
    x1_toks = set([k for k in [token for token in nlp(x[x1_ind])] if not k.is_stop and not k.is_punct])
    x2_toks = set([k for k in [token for token in nlp(x[x2_ind])] if not k.is_stop and not k.is_punct])
    
    # Check if there is any overlap in the non-stop words between the two lines
    return len(x1_toks.union(x2_toks)) > 0

# Function to determine if two lines contain at least two similar words
def containsTwoSimilarWords(x, x1_ind, x2_ind):
    tot_sim = 0
    c = 0.4  # Similarity threshold
    # Tokenize and filter out stop words and punctuation from both lines
    x1_toks = set([k for k in [token for token in nlp(x[x1_ind])] if not k.is_stop and not k.is_punct])
    x2_toks = set([k for k in [token for token in nlp(x[x2_ind])] if not k.is_stop and not k.is_punct])
    
    # Check for similarity between words of both lines and count them
    for val in x1_toks:
        for val2 in x2_toks:
            if val.similarity(val2) > c:
                tot_sim += 1
                if tot_sim == 2:
                    return True
    return False

# Function to determine if two lines possibly have the same metrical length
def possSameMeter(x, x1_ind, x2_ind):
    # Calculate the metrical length of each line by summing the syllables in each word
    x1_len = sum([len(hyphenate.hyphenate_word(i)) for i in x[x1_ind].split(" ")])
    x2_len = sum([len(hyphenate.hyphenate_word(i)) for i in x[x2_ind].split(" ")])
    
    # Check if the metrical lengths of both lines are within a certain range
    return abs(x1_len - x2_len) <= 2

# Function to determine if two lines end with the same word
def sameEndWord(x, x1_ind, x2_ind):
    return word(x[x1_ind].split(" ")[-1].lower()) in x[x2_ind].lower() or word(x[x2_ind].split(" ")[-1].lower()) in x[x1_ind].lower()
