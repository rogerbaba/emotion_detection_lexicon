import string
pos_map = {} # not used!
pos_map["NN"] = "n"
pos_map["NNP"] = "n"
pos_map["NNS"] = "n"
pos_map["NNPS"] = "n"
pos_map["NNP"] = "n"
pos_map["VB"] = "v"
pos_map["VBD"] = "v"
pos_map["VBG"] = "v"
pos_map["VBN"] = "v"
pos_map["VBP"] = "v"
pos_map["VBZ"] = "v"
pos_map["JJ"] = "a"
pos_map["JJR"] = "a"
pos_map["JJS"] = "a"
pos_map["RB"] = "r"
pos_map["RBR"] = "r"
pos_map["RBS"] = "r"

def nltk_tag_to_wordnet_tag(nltk_tag):
    from nltk.corpus import wordnet
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def remove_stopwords(word_tokens):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def build_bow(sentence: str):
    """
    Build a Bag of words from a sentence
    """
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])
    return bow

def print_bow(bow: {}):
    """
    Print a bag of words
    :param bow: A bag od words to print
    :return:
    """
    print(f"Bag of word sentence 1 :\n{bow}")

def tokenize(sentence: str):
    from nltk.tokenize import word_tokenize
    result_tokens = word_tokenize(sentence)
    return result_tokens

def lemm_tokens(word_tokens):
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    pos_tokens = pos_tag(word_tokens)
    print(pos_tokens)
    result_tokens = []
    # Map to
    # first element is the word, second element is the pos tag
    for token in pos_tokens :

        # skip not mapped speech
        if token[1].upper() != token[0].upper():
            # Get Part of Speech from pos_tag to relevant tag of the lemmatizer
            # example 'NNP' => 'n'
            pos = nltk_tag_to_wordnet_tag(token[1])
        else:
            pos = ""
        if pos != "" and pos is not None:
            result_tokens.append(lemmatizer.lemmatize(token[0], pos=pos))
        else:
             result_tokens.append(token[0])
    return result_tokens

def get_emotions(tokens):
    from senticnet.senticnet import SenticNet
    result = {}
    sn = SenticNet()
    for token in tokens:
        moodtags = ""

        if token in sn.data:
            moodtags = sn.moodtags(token)
            print(token, moodtags)
    #TODO
    return result

# inti corpora
sentence = "Could you please not take this? John does not likes to watch movies. Mary does not likes movies from Austria. :) (see under http://rogers.li) #lovemovie N"
sentence2 = "John like watch movies . Mary likes movies . happy"
print(sentence)
# processing:
# remove stopwords -> remove special string/words -> Get POS -> Lemm -> lookup emotion for each word -> w emotions
#
# Create Tokens
print("Create Tokens:")
tokens1 = tokenize(sentence)
print(tokens1)

# lemmatize tokens
print("Lemm Tokens")
tokens2 = lemm_tokens(tokens1)
print(tokens2)

# Remove stopwords
print("Remove Stopwords")
tokens3 = remove_stopwords(tokens2)
#sentence2 = sentence
print(' '.join(tokens3))
#print_bow(build_bow([sentence]))

# get the emotions tag of the tokens from senticnet
print("===============Emotions=====================")
get_emotions(tokens3)
