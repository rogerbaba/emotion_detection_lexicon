#
#  Lexicon Based emotion detection in text corpora.
#
#  Roger Barras
#

text = 'I love to read books. Because I enjoy it.'
text = 'watch'

# Bag of Concepts

import bagofconcepts as boc


# Each line of corpus must be equivalent to each document of the corpus
#boc_model=boc.BOCModel(doc_path="input corpus path")
boc_model=boc.BOCModel('text.txt')

#boc_model.context = text

# output can be saved with save_path parameter
boc_matrix,word2concept_list,idx2word_converter=boc_model.fit()



# SenitcNet lexicon lookup
from senticnet.senticnet import SenticNet

sn = SenticNet()

concept_info = sn.concept(text)
polarity_value = sn.polarity_value(text)
polarity_intense = sn.polarity_intense(text)
moodtags = sn.moodtags(text)
semantics = sn.semantics(text)
sentics = sn.sentics(text)

print('==================================')
print('test: ',text)
print('concept_info: ',concept_info)
print('polarity_value: ', polarity_value)
print('polarity_intense: ',polarity_intense)
print('moodtags: ',moodtags)
print('semantics: ',semantics)
print('sentics: ',sentics)
print('==================================')

