
import wandb
from wandb_config import config; config()

sweep_config = {
    'name': 'bert-base-uncased',
    'program': 'train.py',
    'method': 'grid',
    'parameters': {
        'pooling': {
            'values': ['max', 'average'],
        },
        'submodel': {
            # 'values': [
            #     'word_embedding,0,1,2,3,4,5,6,7,8,9,10,11,12',
            #     'word_embedding,0,5,6,11,12',
            #     'word_embedding,0,1,2,3,4',
            #     '7,8,9,10,11,12',
            # ],
            'values': ['word_embedding'] + list(range(13)),
        },
    },
    'command': [
        '${env}',
        '${interpreter}',
        '${program}',
        '${args}',
        '--mode=lm+head',
        '--lm=gpt2',
        '--cv_splits=2',
        '--epochs=50',
        '--head_cmd=MLP(input_dim, 16, output_dim, dropout=0.5, norm=True, norm_momentum=0.1)',
        "--loss_cmd=MSELoss(reduction='none')",
        '--lr=0.0001',
        '--wandb_on',
    ],
}

sweep_id = wandb.sweep(sweep_config)
print(sweep_id)