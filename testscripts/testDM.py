import pandas as pd
import numpy as np
import tqdm
# Test
def read_depechemood():
    dm_url = "https://raw.githubusercontent.com/marcoguerini/DepecheMood/master/DepecheMood%2B%2B/DepecheMood_english_token_full.tsv"
    depechemood = pd.read_csv(dm_url, sep='\t', index_col=0)
    depechemood = depechemood[depechemood['freq']> 10]
    depechemood.drop('freq', inplace=True, axis=1)
    return depechemood

def extract_emofeats(lex, lex_vocab, text):
    lex_dict = lex.to_dict('split')
    lex_d = {word: lex_dict['data'][i] for i, word in enumerate(lex_dict['index'])}
    assert len(lex_d.keys()) == len(lex_vocab)
    Ss = np.zeros((len(text), lex.shape[1]))
    for i, doc in tqdm(enumerate(text)):
        intersection = lex_vocab & set(doc)
        s = []
        for inter in intersection:
            s.append(lex_d[inter])
        s = np.array(s)
        divisor = len(s) if len(s) > 0 else 1
        Ss[i, :] = np.sum(s, axis=0) / divisor
    return Ss

# Example of use
texts = [
    ['my', 'dog', 'is', 'happy'],
    ['my', 'cat', 'is', 'sad'],
]
dm = read_depechemood()
features = extract_emofeats(dm, set(dm.index.values), texts[0])
#features = extract_emofeats(dm, set(dm.index.values), comments['body_processed'].str.split(' '))