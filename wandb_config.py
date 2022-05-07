import os
import getpass
import wandb

wandb_dir = f'/tmp/wandb_{getpass.getuser()}'
os.makedirs(wandb_dir, exist_ok=True)

settings = {
    'WANDB_ENTITY': 'duxin',    # replace this with your WANDB account name
    'WANDB_API_KEY': '4cd046a5edd8dd0a58a2060aa5177f4639871e60',   # replace this with your WANDB API KEY
    'WANDB_DIR': wandb_dir,
    'WANDB_PROJECT': 'readability',  # you can change this to the name you like
    'WANDB_BASE_URL': 'https://api.wandb.ai/',
}

def config():
    for k, v in settings.items():
        os.environ[k] = v