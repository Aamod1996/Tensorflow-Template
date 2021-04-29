import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from data_loader.generator import Generator
from models.model import CharacterModel
from trainers.trainer import Trainer
from utils.config import process_config
from utils.utils import get_args


def get_ckpt_path(config):
    ckpt_path = "checkpoints/" + config.name
    
    if os.path.exists(ckpt_path):
        return True
    else:
        return None
    
def main():

    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("Missing or invalid arguments")
        exit(0)

    generator = Generator(config)
    output_dims = generator.max_id
    
    model = CharacterModel(output_dims, config)
    
    if get_ckpt_path(config):
        print("Weights found. Loading pretrained model...\n")
        model.load_params()
        
    trainer = Trainer(model, generator, config)
    
    trainer.train()

if __name__ == '__main__':
    main()