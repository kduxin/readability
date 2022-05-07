import json
import tqdm
import transformers
import spacy
nlp = spacy.load('en_core_web_sm')
import sys; sys.path.append('..')

def load_data(path):
    import pandas as pd
    df = pd.read_csv(path)
    df['excerpt'] = df['excerpt'].str.lower()
    df = df.loc[df['standard_error'] > 0]
    return df

df = load_data('../data/commonlitreadabilityprize/train.csv')
print('Texts loaded.')

lower = True

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

syntdata = []
for i, sent, score in zip(tqdm.tqdm(df['id']), df['excerpt'], df['target']):

    if lower:
        sent = sent.lower()
    tokens = tokenizer.tokenize(sent)
    ntokens = len(tokens)

    # spaCy predictor
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    doc = nlp(doc)
    pos_tags = [token.tag_ for token in doc]

    dep_head = [0 for _ in range(ntokens)]
    for token in doc:
        for c in token.children:
            assert dep_head[c.i] == 0
            dep_head[c.i] = token.i + 1
    dep_label = [token.dep_.lower() for token in doc]


    syntsample = {
        'sentence_id': i,
        'tokens': tokens,
        'pos_tags': pos_tags,
        'verb_indicator': [None, tokens],
        'ontonotes_head': dep_head,
        'ontonotes_deprel': dep_label,
        'tags': [None, tokens],
        'metadata': {
            'words': tokens,
            'verb': None,
            'verb_index': None,
            'gold_tags': None,
        }
    }
    
    syntdata.append(syntsample)

save_path = '../data/dep/train.json'
with open(save_path, 'wt') as f:
    for syntsample in syntdata:
        f.write(json.dumps(syntsample) + '\n')