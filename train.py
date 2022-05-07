
import os
import re
import random
import string
import copy
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
import transformers
import torch
from torch import nn
from torch.nn import (
    functional as F,
    MSELoss,
)
from typing import List
from argparse import Namespace

import wandb
from wandb_config import config; config()


# XGBOOST
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBRegressor # <3
import gc


# Syntax-Augmented BERT
from lib.syntax_augmented_bert.utils.loader import (
    FeaturizedDataset,
    FeaturizedDataLoader,
)
from lib.syntax_augmented_bert import model
from lib.syntax_augmented_bert.utils import (
    constant,
)
from lib.syntax_augmented_bert.utils.utils import (
    OntoNotesSRLProcessor,
    InputExample,
)
from lib.syntax_augmented_bert.opt import get_args
from lib.syntax_augmented_bert.main import (
    load_and_cache_examples,
    MODEL_CLASSES,
)




# ========================================================
#                          data 
# ========================================================

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df['excerpt'] = df['excerpt'].str.lower()
    df = df.loc[df['standard_error'] > 0]
    return df

def load_syntax_data(path) -> List[InputExample]:
    examples = OntoNotesSRLProcessor().get_unk_examples(path)
    return examples

# ========================================================
#                          utils 
# ========================================================
def seed_everything(seed=1234):
    import os
    import random
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True





# ========================================================
#                           model
# ========================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, activation='sigmoid', norm=True, norm_momentum=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(hidden_dim, momentum=norm_momentum) if norm else None
        self.act = {
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
        }[activation]
        
    def forward(self, x):
        x = self.fc1(x)
        if callable(self.norm):
            x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


class LayerMixer(nn.Module):
    def __init__(self, num_layers):
        super(LayerMixer, self).__init__()
        self.a = nn.Parameter(torch.randn(num_layers, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    @property
    def weight(self):
        return torch.softmax(self.a, dim=0)

    def forward(self, x):
        """mixing

        Args:
            x (List[Tensor(batch, dim)])

        Returns:
            mix (Tensor(batch, dim)) 
        """
        return self.gamma * (torch.stack(x, dim=-1) * self.weight).sum(dim=-1)
        


# ========================================================
#                      train & eval & test 
# ========================================================


def do_pred(lm, head, tokenizer, X, submodel, device, pooling, batch_size=99999, mixer=None):
    n = len(X)
    preds = []
    for i in range(0, n, batch_size):
        X_batch = X[i:i+batch_size]
        tokens = tokenizer(X_batch, return_tensors='pt', padding=True).to(device)
        textemb = _get_lm_submodal_textemb(lm, tokens, submodel=submodel, pooling=pooling, concat=(mixer is None))
        if mixer is not None:
            textemb = mixer(textemb)
        pred = head(textemb)
        pred = pred.squeeze(-1)
        preds.append(pred)
    if len(preds) == 1:
        preds = preds[0]
    else:
        preds = torch.cat(preds, dim=0)
    return preds

def do_syntaxlm_pred(lm, head, tokenizer, X_synt: FeaturizedDataset,
                     submodel, device, pooling, batch_size=99999, mixer=None):
    n = len(X_synt)
    preds = []
    dataloader = FeaturizedDataLoader(X_synt, opt=Namespace(task_name='readability', device=device), batch_size=batch_size)
    for i in range(0, n, batch_size):
        batch = dataloader._collate_fn(X_synt[i:i+batch_size])
        tokens = {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'wp_token_mask': batch['wp_token_mask'],
            'dep_head': batch['dep_head'],
            'dep_rel': batch['dep_rel'],
            'wp_rows': batch['wp_rows'],
            'align_sizes': batch['align_sizes'],
            'seq_len': batch['seq_len'],
            'subj_pos': batch['subj_pos'],
            'obj_pos': batch['obj_pos'],
            'linguistic_token_mask': batch['linguistic_token_mask'],
        }

        textemb = _get_lm_submodal_textemb(lm, tokens, submodel=submodel, pooling=pooling, concat=(mixer is None))
        if mixer is not None:
            textemb = mixer(textemb)
        pred = head(textemb)
        pred = pred.squeeze(-1)
        preds.append(pred)
    if len(preds) == 1:
        preds = preds[0]
    else:
        preds = torch.cat(preds, dim=0)
    return preds

def _get_lm_submodal_textemb(lm, tokens, submodel, pooling, concat=True):

    def _pooling(hidden_states, mask):
        if pooling == 'average':
            textemb = hidden_states.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        elif pooling == 'max':
            textemb = (hidden_states * mask.unsqueeze(-1)).max(dim=1).values
        else:
            raise ValueError(pooling)
        return textemb
        

    num_layers = guess_num_layers(lm)

    segs = re.split(r'[,-]', submodel)
    if any([seg != 'word_embedding' for seg in segs]):
        output = lm(**tokens, output_hidden_states=True)

    if hasattr(tokens, 'attention_mask'):
        mask = tokens.attention_mask
    elif isinstance(tokens, dict) and ('wp_token_mask' in tokens):
        mask = tokens['wp_token_mask']
    else:
        raise ValueError

    if isinstance(tokens, dict):
        input_ids = tokens['input_ids']
    else:
        input_ids = tokens.input_ids
        
    textembs = []
    for layer in segs:
        if layer.isdecimal():
            layer = int(layer)
            # assert 0 <= layer <= num_layers
            hidden_state = output[-1][layer] # (batch_size, seqlen, dim)
            textemb = _pooling(hidden_state, mask)
        
        else:
            assert layer in ['word_embedding', 'pooler']
            if layer == 'word_embedding':
                wordemb = guess_wordemb(lm)
                hidden_state = wordemb(input_ids)
                textemb = _pooling(hidden_state, mask)
            elif layer == 'pooler':
                textemb = output[1]
            else:
                raise ValueError(layer)
        
        textembs.append(textemb)
    
    if '-' in submodel:
        # subtract two layers
        assert len(textembs) == 2
        return textembs[0] - textembs[1]
    else:
        if concat:
            return torch.cat(textembs, dim=-1)
        else:
            return textembs


def train_lm_readeval(lm, head, tokenizer, loss_func, X, y_mean, y_std, args, eval_X=None, eval_y_mean=None, eval_y_std=None, mixer=None, syntax=False):
    if isinstance(X, pd.Series):
        X = X.tolist()

    # split X into train set + eval set
    num_train = int(args.cv_ratio_train * len(X))
    X_train, X_eval = X[:num_train], X[num_train:]
    y_mean_train, y_mean_eval = y_mean[:num_train], y_mean[num_train:]
    y_std_train, y_std_eval = y_std[:num_train], y_std[num_train:]

    if syntax:
        pred_func = do_syntaxlm_pred
    else:
        pred_func = do_pred

    # construct readability evaluation model
    head = copy.deepcopy(head)
    parameters = [{
        'params': head.parameters(),
    }]
    if mixer is not None:
        mixer = copy.deepcopy(mixer)
        parameters += [{
            'params': mixer.parameters(),
            'lr': 1e-2,
        }]
    if syntax:
        parameters += [{
            'params': lm.syntax_encoder.parameters(),
            'lr': args.syntax_encoder_lr,
        }]
    if args.train_lm:
        lm = copy.deepcopy(lm)
        lm.train()
        parameters += [{
            'params': lm.parameters()
        }]
    else:
        lm.eval()
    lm.requires_grad_(args.train_lm)
    if syntax:
        lm.syntax_encoder.requires_grad_(True)
        lm.syntax_encoder.train()
    print("Parameters:")
    print(parameters)

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
    else:
        raise ValueError(args.optimizer)
    
    best_corr = - float('inf')
    best_model = {'head': head}
    if args.train_lm:
        best_model['lm'] = lm
    if mixer is not None:
        best_model['mixer'] = mixer
    # train
    for i in range(args.epochs):
        head.train()
        for j in range(0, len(X_train), args.batch_size):
            X_train_batch = X_train[j : j+args.batch_size]
            y_mean_train_batch = torch.tensor(y_mean_train[j : j+args.batch_size], dtype=torch.float32, device=args.device)
            y_std_train_batch = torch.tensor(y_std_train[j : j+args.batch_size], dtype=torch.float32, device=args.device)

            optimizer.zero_grad()
            pred = pred_func(lm, head, tokenizer, X_train_batch,
                             submodel=args.submodel, device=args.device, pooling=args.pooling, mixer=mixer)

            loss = loss_func(pred, y_mean_train_batch)
            mean_loss = loss.mean()
            mean_loss.backward()
            optimizer.step()



        # eval
        with torch.no_grad():
            head.eval()
            # losses = []
            # preds = []
            # for j in range(0, len(X_eval), args.batch_size):
            #     X_eval_batch = X_eval[j : j+args.batch_size]
            #     y_mean_eval_batch = torch.tensor(y_mean_eval[j : j+args.batch_size], dtype=torch.float32, device=args.device)
            #     y_std_eval_batch = torch.tensor(y_std_eval[j : j+args.batch_size], dtype=torch.float32, device=args.device)
            #     pred = do_pred(lm, head, tokenizer, X_eval_batch,
            #                    submodel=args.submodel, device=args.device, pooling=args.pooling)
            #     pred = pred.squeeze(dim=-1)
            #     preds.extend(pred.tolist())
            #     loss = loss_func(pred, y_mean_eval_batch)
            #     losses.extend(loss.tolist())
            
            pred = pred_func(lm, head, tokenizer, X_eval,
                             submodel=args.submodel, device=args.device, pooling=args.pooling,
                             batch_size=args.batch_size,
                             mixer=mixer)
            loss = loss_func(pred, torch.tensor(y_mean_eval, dtype=torch.float32, device=args.device))
            loss = loss.tolist()
            eval_mean_loss = sum(loss) / len(loss)
            corr = pd.DataFrame({'pred': pred.tolist(), 'label': y_mean_eval}).corr().loc['pred', 'label']
            print(f'Epoch {i+1:0>2}. mean_loss={eval_mean_loss:.3f}. corr={corr:.3f}.')
            print(f'Std of pred: {pred.std().item():.3f}')
            if mixer is not None:
                print(f'Mixer weights: {mixer.weight}')

            if corr > best_corr:
                best_model['head'] = copy.deepcopy(head)
                if args.train_lm:
                    best_model['lm'] = copy.deepcopy(lm)
                if mixer is not None:
                    best_model['mixer'] = copy.deepcopy(mixer)
                best_corr = corr

            # if args.wandb_on:
            #     wandb.log({
            #             f'{log_prefix}loss/train': mean_loss,
            #             f'{log_prefix}loss/eval': eval_mean_loss,
            #             f'{log_prefix}corr/eval': corr,
            #         }, commit=False)
        
        if (eval_X is not None) and (eval_y_mean is not None) and (eval_y_std is not None):
            pred = pred_func(lm, head, tokenizer, eval_X, submodel=args.submodel, device=args.device, pooling=args.pooling, mixer=mixer)
            corr = pd.DataFrame({'pred': pred.tolist(), 'label': eval_y_mean}).corr().loc['pred', 'label']
            print(f"Correlation on evaluation set is {corr}")

    return best_model, best_corr




# def test_lm_readeval(lm, head, tokenizer, loss_func, X, y_mean, y_std, args):
#     # split X into train set + eval set
#     if isinstance(X, pd.Series):
#         X = X.tolist()
    
#     # eval
#     with torch.no_grad():
#         lm.eval()
#         head.eval()
#         preds = []
#         losses = []
#         for j in range(0, len(X), args.batch_size):
#             X_eval_batch = X[j : j+args.batch_size]
#             y_mean_eval_batch = y_mean[j : j+args.batch_size]
#             y_std_eval_batch = y_std[j : j+args.batch_size]
#             # pred = head(textemb)  # (batch_size, 1)
#             pred = do_pred(lm, head, tokenizer, X_eval_batch,
#                            submodel=args.submodel, device=args.device, pooling=args.pooling)
#             # pred = pred.squeeze(dim=-1)
#             loss = loss_func(pred, y_mean_eval_batch, y_std_eval_batch)
#             losses.extend(loss.tolist())
#             preds.extend(pred.tolist())
#         # print(pred)
#     return losses, preds

def _preprocess_text(textseries, lower=True):
    return textseries.str.lower().tolist()

def guess_wordemb(lm):
    try:
        return lm.embeddings.word_embeddings
    except:
        pass

    try:
        return lm.wte
    except:
        pass

    raise ValueError(f'Failed to guess word embedding.')

def guess_lm_inputdim(lm, lm_name):
    if 'bert' in lm_name:
        return lm.embeddings.word_embeddings.embedding_dim
    
    elif 'gpt2' in lm_name:
        return lm.wte.embedding_dim
    
    raise ValueError(f'Failed to guess input dimensionality of {lm_name}.')

def guess_num_layers(lm, lm_name=None):
    if lm_name is not None:
        if 'bert' in lm_name:
            return len(lm.encoder.layer)
        
        elif 'gpt2' in lm_name:
            return len(lm.h)
    
    else:
        try:
            return len(lm.encoder.layer)
        except:
            pass
            
        try:
            return len(lm.h)
        except:
            pass

    raise ValueError(f'Failed to guess number of the hidden layers of {lm_name}.')


def main(args):

    args.name = name = ''.join(random.choices(string.ascii_uppercase, k=8))
    seed_everything(args.seed)
    # update args
    args.device = device = 'cpu' if args.cpu else 'cuda'

    if '-' in args.submodel:
        left, right = args.submodel.split('-')
        assert len(left.split(',')) == len(right.split(',')) == 1
        args.dim_multiplier = 1
    elif args.mixing:
        args.dim_multiplier = 1
    else:
        args.dim_multiplier = len(args.submodel.split(','))

    args.savedir = savedir = os.path.join(os.path.abspath(args.savedir), name)
    os.makedirs(args.savedir, exist_ok=True)
    print(f'Savedir = {savedir}')

    if args.test:
        args.epochs = 5

    # wandb init, upload args
    if args.wandb_on:
        wandb.init()
        wandb.log(args.__dict__)

    if args.mode in ['lm+head']:
        main_lm_head(args)
    elif args.mode in ['syntaxlm+head']:
        main_syntaxlm_head(args)
    elif args.mode == 'tfidf+xgboost':
        main_tfidf_xgboost(args)
    

def main_tfidf_xgboost(args):
    data = load_data(args.dataset)

    X, y_mean, y_std = data['excerpt'], data['target'], data['standard_error']
    
    print("TFIDF")
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        norm='l2',
        min_df=0,
        smooth_idf=False,
        max_features=15000)
    word_vectorizer.fit(X)

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        norm='l2',
        min_df=0,
        smooth_idf=False,
        max_features=50000)
    char_vectorizer.fit(X)

    
    cv_corr, cv_preds, cv_labels = [], [], []
    for i, (train_ids, test_ids) in enumerate(
            sklearn.model_selection.KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed).split(data)
            ):
        print(f'\n=================== cross validation {i+1} / {args.cv_splits} =================')
        data_train, data_test = data.iloc[train_ids], data.iloc[test_ids]
        X_train, y_train_mean, y_train_std = data_train['excerpt'], data_train['target'], data_train['standard_error']
        X_test, y_test_mean, y_test_std    = data_test['excerpt'], data_test['target'], data_test['standard_error']
        X_train = _preprocess_text(X_train, lower=True)
        X_test  = _preprocess_text(X_test, lower=True)

        train_word_features = word_vectorizer.transform(X_train)
        test_word_features = word_vectorizer.transform(X_test)
        train_char_features = char_vectorizer.transform(X_train)
        test_char_features = char_vectorizer.transform(X_test)

        train_features = hstack([train_char_features, train_word_features])
        test_features = hstack([test_char_features, test_word_features])

        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(train_features, y_train_mean)

        # if args.wandb_on:
        #     wandb.log({f'corr/cv{i:0>2d}/eval_best': None})

        # ============= save =============
        path = f'{args.savedir}/cv{i:0>2d}_xgb'
        torch.save(model, path)
        print(f'Saved XGBRegressor to {path}')
        if args.wandb_on:
            wandb.save(path, base_path=args.savedir, policy='now')
            
        if args.test:
            break


        # ============= cross validation test =============
        pred = model.predict(test_features)
        corr = pd.DataFrame({'pred': pred, 'label': y_test_mean}).corr().loc['pred', 'label']
        print(f'corr={corr:.3f}')
        cv_preds.extend(pred.tolist())
        cv_labels.extend(y_test_mean.tolist())
        cv_corr.append(corr)
    
    if args.wandb_on:
        mean_corr = np.mean(cv_corr)
        overall_corr = pd.DataFrame({'pred': cv_preds, 'label': cv_labels}).corr().loc['pred', 'label']
        print(f'mean_corr={mean_corr:.3f}. overall_corr={overall_corr:.3f}')
        wandb.log({
            'mean_corr': mean_corr,
            'overall_corr': overall_corr,
        })


def main_syntaxlm_head(args):

    assert args.lm in ['bert-base-uncased']

    # load data, tokenizer
    data: pd.DataFrame = load_data(args.dataset)
    syntax_data: List[InputExample] = load_syntax_data(args.syntax_dataset)
    print(f'Length of data = {len(data)}')
    print(f'Length of syntax_data = {len(syntax_data)}')

    synt_opt = get_args([
        '--model_type=syntax_bert_seq',
        '--model_name_or_path=bert-base-uncased',
        '--task_name=ontonotes_srl',
        '--data_dir=data/dep/',
        '--max_seq_length=512',
        '--per_gpu_eval_batch_size=32',
        '--output_dir=results/syntax/checkpoints/1/'
        '--save_steps=1000', 
        '--overwrite_output_dir',
        '--num_train_epochs=20',
        '--do_eval',
        '--do_train',
        '--evaluate_during_training',
        '--config_name_or_path=lib/syntax_augmented_bert/config/srl/bert-base-uncased/joint_fusion.json',
        '--per_gpu_train_batch_size=16',
        '--gradient_accumulation_steps=1',
        '--wordpiece_aligned_dep_graph',
        '--seed=40',
    ])
    synt_label_map = constant.OntoNotes_SRL_LABEL_TO_ID
    synt_num_labels = len(synt_label_map)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    cv_corr, cv_preds, cv_labels = [], [], []
    for i, (train_ids, test_ids) in enumerate(
            sklearn.model_selection.KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed).split(data)
            ):
        print(f'\n=================== cross validation {i+1} / {args.cv_splits} =================')
        data_train, data_test = data.iloc[train_ids], data.iloc[test_ids]
        X_train_synt = FeaturizedDataset(
            examples=[syntax_data[idx] for idx in train_ids],
            opt=synt_opt,
            tokenizer=tokenizer,
            label_map=synt_label_map,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
        )
        X_test_synt = FeaturizedDataset(
            examples=[syntax_data[idx] for idx in test_ids],
            opt=synt_opt,
            tokenizer=tokenizer,
            label_map=synt_label_map,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
        )

        X_train, y_train_mean, y_train_std = data_train['excerpt'], data_train['target'], data_train['standard_error']
        X_test, y_test_mean, y_test_std    = data_test['excerpt'], data_test['target'], data_test['standard_error']
        X_train = _preprocess_text(X_train, lower=True)
        X_test  = _preprocess_text(X_test, lower=True)

        synt_config = model.SyntaxBertConfig.from_pretrained(f'lib/syntax_augmented_bert/config/srl/bert-base-uncased/{args.syntax_fusion}.json',
                                                             num_labels=synt_num_labels,
                                                             finetuning_task='ontonotes_srl')
        syntlm = model.SyntaxBertForSequenceClassification.from_pretrained(args.lm, config=synt_config)
        lm = syntlm.bert.to(args.device)
        lm.requires_grad_(args.train_lm)
        lm.syntax_encoder.requires_grad_(True)
        input_dim = guess_lm_inputdim(lm, args.lm) * args.dim_multiplier
        output_dim = 1

        head      = eval(args.head_cmd).to(args.device)
        mixer     = LayerMixer(num_layers=len(re.split(r'[-,]', args.submodel))).to(args.device) if args.mixing else None
        loss_func = eval(args.loss_cmd).to(args.device)

        additional_kwargs = {}
        if args.test:
            additional_kwargs.update({
                'eval_X': X_test,
                'eval_y_mean': y_test_mean,
                'eval_y_std': y_test_std
                })
        if args.mixing:
            additional_kwargs.update({
                'mixer': mixer,
            })
        
        trained_models, best_corr = train_lm_readeval(lm, head, tokenizer, loss_func, X_train_synt, y_train_mean, y_train_std, args, syntax=True, **additional_kwargs)
        if args.wandb_on:
            wandb.log({f'corr/cv{i:0>2d}/eval_best': best_corr})

        # ============= save =============
        if args.train_lm:
            path = f'{args.savedir}/cv{i:0>2d}_lm'
            torch.save(trained_models['lm'], path)
            print(f'Saved lm to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
        if True:
            path = f'{args.savedir}/cv{i:0>2d}_head'
            torch.save(trained_models['head'], path)
            print(f'Saved head to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
        if args.mixing:
            path = f'{args.savedir}/cv{i:0>2d}_mixer'
            mixer = trained_models["mixer"]
            torch.save(mixer, path)
            print(f'Mixer weights: {mixer.weight}')
            print(f'Saved mixer to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
            
            
        if args.test:
            break

        # ============= cross validation test =============
        with torch.no_grad():
            lm = trained_models.get('lm', lm)
            lm.eval()
            head = trained_models.get('head', head)
            head.eval()
            pred = do_syntaxlm_pred(
                lm=lm, head=head, tokenizer=tokenizer,
                X_synt=X_test_synt,
                submodel=args.submodel, device=args.device, pooling=args.pooling,
                batch_size=args.batch_size, mixer=mixer)
        pred = pred.tolist()
        cv_preds.extend(pred)
        cv_labels.extend(y_test_mean.tolist())
        corr = pd.DataFrame({'pred': pred, 'label': y_test_mean}).corr().loc['pred', 'label']
        cv_corr.append(corr)
    
    if args.wandb_on:
        mean_corr = np.mean(cv_corr)
        overall_corr = pd.DataFrame({'pred': cv_preds, 'label': cv_labels}).corr().loc['pred', 'label']
        print(f'mean_corr={mean_corr:.3f}. overall_corr={overall_corr:.3f}')
        if args.mixing:
            mixer_weight_log = {}
            for i, seg in enumerate(re.split(r'[-,]', args.submodel)):
                weight = trained_models['mixer'].weight
                mixer_weight_log[f'mixer_weight/{seg:0>2}'] = weight[i].item()
            wandb.log(mixer_weight_log)
        wandb.log({
            'mean_corr': mean_corr,
            'overall_corr': overall_corr,
        })



def main_lm_head(args):
    # load data, tokenizer
    data: pd.DataFrame = load_data(args.dataset)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    cv_corr, cv_preds, cv_labels = [], [], []
    for i, (train_ids, test_ids) in enumerate(
            sklearn.model_selection.KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed).split(data)
            ):
        print(f'\n=================== cross validation {i+1} / {args.cv_splits} =================')
        data_train, data_test = data.iloc[train_ids], data.iloc[test_ids]
        X_train, y_train_mean, y_train_std = data_train['excerpt'], data_train['target'], data_train['standard_error']
        X_test, y_test_mean, y_test_std    = data_test['excerpt'], data_test['target'], data_test['standard_error']
        X_train = _preprocess_text(X_train, lower=True)
        X_test  = _preprocess_text(X_test, lower=True)

        lm = transformers.AutoModel.from_pretrained(args.lm).to(args.device)
        lm.requires_grad_(args.train_lm)
        input_dim = guess_lm_inputdim(lm, args.lm) * args.dim_multiplier
        output_dim = 1

        head      = eval(args.head_cmd).to(args.device)
        mixer     = LayerMixer(num_layers=len(re.split(r'[-,]', args.submodel))).to(args.device) if args.mixing else None
        loss_func = eval(args.loss_cmd).to(args.device)

        additional_kwargs = {}
        if args.test:
            additional_kwargs.update({
                'eval_X': X_test,
                'eval_y_mean': y_test_mean,
                'eval_y_std': y_test_std
                })
        if args.mixing:
            additional_kwargs.update({
                'mixer': mixer,
            })
        
        trained_models, best_corr = train_lm_readeval(lm, head, tokenizer, loss_func, X_train, y_train_mean, y_train_std, args, **additional_kwargs)
        if args.wandb_on:
            wandb.log({f'corr/cv{i:0>2d}/eval_best': best_corr})

        # ============= save =============
        if args.train_lm:
            path = f'{args.savedir}/cv{i:0>2d}_lm'
            torch.save(trained_models['lm'], path)
            print(f'Saved lm to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
        if True:
            path = f'{args.savedir}/cv{i:0>2d}_head'
            torch.save(trained_models['head'], path)
            print(f'Saved head to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
        if args.mixing:
            path = f'{args.savedir}/cv{i:0>2d}_mixer'
            mixer = trained_models["mixer"]
            torch.save(mixer, path)
            print(f'Mixer weights: {mixer.weight}')
            print(f'Saved mixer to {path}')
            if args.wandb_on:
                wandb.save(path, base_path=args.savedir, policy='now')
            
            
        if args.test:
            break

        # ============= cross validation test =============
        with torch.no_grad():
            lm = trained_models.get('lm', lm)
            lm.eval()
            head = trained_models.get('head', head)
            head.eval()
            pred = do_pred(lm=lm, head=head,
                           tokenizer=tokenizer, X=X_test,
                           submodel=args.submodel, device=args.device, pooling=args.pooling,
                           batch_size=args.batch_size, mixer=mixer)
        pred = pred.tolist()
        cv_preds.extend(pred)
        cv_labels.extend(y_test_mean.tolist())
        corr = pd.DataFrame({'pred': pred, 'label': y_test_mean}).corr().loc['pred', 'label']
        cv_corr.append(corr)
    
    if args.wandb_on:
        mean_corr = np.mean(cv_corr)
        overall_corr = pd.DataFrame({'pred': cv_preds, 'label': cv_labels}).corr().loc['pred', 'label']
        print(f'mean_corr={mean_corr:.3f}. overall_corr={overall_corr:.3f}')
        if args.mixing:
            mixer_weight_log = {}
            for i, seg in enumerate(re.split(r'[-,]', args.submodel)):
                weight = trained_models['mixer'].weight
                mixer_weight_log[f'mixer_weight/{seg:0>2}'] = weight[i].item()
            wandb.log(mixer_weight_log)
        wandb.log({
            'mean_corr': mean_corr,
            'overall_corr': overall_corr,
        })


def get_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset',        type=str,   default='data/commonlitreadabilityprize/train.csv')
    parser.add_argument('--savedir',        type=str,   default='results/')
    parser.add_argument('--mode',           type=str,   default='lm+head',
                        choices=['lm+head', 'tfidf+xgboost', 'syntaxlm+head'])
    parser.add_argument('--lm',             type=str,   default='bert-base-uncased')
    parser.add_argument('--syntax_fusion',  type=str,   default='joint_fusion',
                        choices=['late_fusion', 'joint_fusion'])
    parser.add_argument('--syntax_dataset', type=str,   default='data/dep/train.json')
    parser.add_argument('--seed',           type=int,   default=0)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--cv_splits',      type=int,   default=2)
    parser.add_argument('--cv_ratio_train', type=float, default=0.8)
    parser.add_argument('--submodel',       type=str,   default='word_embedding')
    parser.add_argument('--head_cmd',       type=str,   default='MLP(input_dim, 16, output_dim, dropout=0.5, norm=True, norm_momentum=0.1)')
    parser.add_argument('--loss_cmd',       type=str,   default="MSELoss(reduction='none')")
    parser.add_argument('--optimizer',      type=str,   default='AdamW')
    parser.add_argument('--lr',             type=float, default=0.0001)
    parser.add_argument('--syntax_encoder_lr', type=float, default=0.01)
    parser.add_argument('--train_lm',       action='store_true')
    parser.add_argument('--pooling',        type=str,   default='max')
    parser.add_argument('--wandb_on',       action='store_true')
    parser.add_argument('--test',           action='store_true')
    parser.add_argument('--cpu',            action='store_true')
    parser.add_argument('--mixing',         action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())