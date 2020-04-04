#
# Example is creating a simple bag of words from
#
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

punctuation = u",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/"
stop_words_eng = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tag_dict = {"J": wn.ADJ,
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV}

def extract_wnpostag_from_postag(tag):
    #take the first letter of the tag
    #the second parameter is an "optional" in case of missing key in the dictionary
    return tag_dict.get(tag[0].upper(), None)

def lemmatize_tupla_word_postag(tupla):
    """
    giving a tupla of the form (wordString, posTagString) like ('guitar', 'NN'), return the lemmatized word
    """
    tag = extract_wnpostag_from_postag(tupla[1])
    return lemmatizer.lemmatize(tupla[0], tag) if tag is not None else tupla[0]

def bag_of_words(sentence, stop_words=None):
    from nltk import pos_tag
    if stop_words is None:
        stop_words = stop_words_eng
    original_words = word_tokenize(sentence)
    tagged_words = pos_tag(original_words) #returns a list of tuples: (word, tagString) like ('And', 'CC')
    original_words = None
    lemmatized_words = [ lemmatize_tupla_word_postag(ow) for ow in tagged_words ]
    tagged_words = None
    cleaned_words = [ w for w in lemmatized_words if (w not in punctuation) and (w not in stop_words) ]
    lemmatized_words = None
    return cleaned_words

sentence = "Two electric guitar rocks players, and also a better bass player, are standing off to two sides reading corpora while walking"
print(sentence, "\n\n bag of words:\n", bag_of_words(sentence) )