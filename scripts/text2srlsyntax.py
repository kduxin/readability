import json
import tqdm
import spacy
nlp = spacy.load('en_core_web_sm')
import sys; sys.path.append('..')

DEFAULT_SRLMODEL_PATH= "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"

def load_srl_predictor(path=DEFAULT_SRLMODEL_PATH):
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.tagging
    predictor = Predictor.from_path(path, cuda_device=0)
    return predictor


def load_data(path):
    import pandas as pd
    df = pd.read_csv(path)
    df['excerpt'] = df['excerpt'].str.lower()
    df = df.loc[df['standard_error'] > 0]
    return df

srl_predictor = load_srl_predictor()
print('SRL predictor loaded.')

df = load_data('../data/commonlitreadabilityprize/train.csv')
print('Texts loaded.')


test = False
if test:
    path = '../lib/syntax-augmented-bert/datasets/conll2005_srl_udv2/wsj-test.json'
    sentences = []
    for i, line in enumerate(open(path, 'rt')):
        sample = json.loads(line)
        sent = ' '.join(sample['tokens'])
        sentences.append(sent)
    srlresults = srl_predictor.predict_batch_json([{'sentence': sent} for sent in sentences])

else:
    batch_size = 100
    srlresults = []
    for istart in tqdm.tqdm(range(0, len(df), batch_size)):
        sentences = df['excerpt'].iloc[istart:istart+batch_size]
        res = srl_predictor.predict_batch_json([{'sentence': sent} for sent in sentences])
        srlresults.extend(res)

import os
import pickle
srl_path = os.path.abspath('../data/synt/srlresults.pkl')
with open(srl_path, 'wb') as f:
    pickle.dump(srlresults, f)
print(f'SRL results saved to {srl_path}')

syntdata = []
for i, sent, score, srlresult in zip(tqdm.tqdm(df['id']), df['excerpt'], df['target'], srlresults):

    # srl predictor
    tokens = srlresult['words']
    ntokens = len(tokens)
    srlverbs = srlresult['verbs']

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

    for srlverb in srlverbs: # loop over all verbs

        verb = srlverb['verb']
        gold_tags = srlverb['tags']

        verb_indicator = [0 for _ in range(ntokens)]
        if 'B-V' in srlverb['tags']:
            verb_index = srlverb['tags'].index('B-V')
            verb_indicator[verb_index] = 1
        else:
            if verb in tokens:
                verb_index = tokens.index(verb)
                verb_indicator[verb_index] = 1
            else:
                verb_index = None

        syntsample = {
            'sentence_id': i,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'verb_indicator': [verb_indicator, tokens],
            'ontonotes_head': dep_head,
            'ontonotes_deprel': dep_label,
            'tags': [gold_tags, tokens],
            'metadata': {
                'words': tokens,
                'verb': verb,
                'verb_index': verb_index,
                'gold_tags': gold_tags,
            }
        }
        
        syntdata.append(syntsample)

save_path = '../data/synt/train.json'
with open(save_path, 'wt') as f:
    for syntsample in syntdata:
        f.write(json.dumps(syntsample) + '\n')