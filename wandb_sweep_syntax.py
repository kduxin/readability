
import wandb
from wandb_config import config; config()

sweep_config = {
    'name': 'bert-base-uncased',
    'program': 'train.py',
    'method': 'grid',
    'parameters': {
        'pooling': {
            'values': ['average'],
        },
        'submodel': {
            # 'values': [
            #     'word_embedding,0,1,2,3,4,5,6,7,8,9,10,11,12',
            #     'word_embedding,0,5,6,11,12',
            #     'word_embedding,0,1,2,3,4',
            #     '7,8,9,10,11,12',
            # ],
            # 'values': ['word_embedding'] + list(range(13)),
            # 'values': ['word_embedding'] + [3, 6, 9, 12],
            'values': [3, 6, 9, 12],
            # 'values': [12],
        },
        'syntax_encoder_lr': {
            'values': [0.01, 0.1],
        },
        'lr': {
            'values': [0.0001, 0.0005],
        }
    },
    'command': [
        '${env}',
        '${interpreter}',
        '${program}',
        '${args}',
        '--mode=syntaxlm+head',
        '--syntax_fusion=joint_fusion',
        '--lm=bert-base-uncased',
        '--cv_splits=2',
        '--epochs=50',
        '--head_cmd=MLP(input_dim, 16, output_dim, dropout=0.5, norm=True, norm_momentum=0.1)',
        "--loss_cmd=MSELoss(reduction='none')",
        '--lr=0.0001',
        '--wandb_on',
        # '--train_lm',
        # '--batch_size=16',
    ],
}

sweep_id = wandb.sweep(sweep_config)
print(sweep_id)