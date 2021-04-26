import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from data_loader.generator import TemplateGenerator
from models.model import TemplateModel
from trainers.trainer import TemplateTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def get_ckpt_path():
    ckpt_path = "checkpoints/"
    
    if os.path.exists(ckpt_path) and not os.path.isfile(ckpt_path):
        if not os.listdir(path):
            return None
        else:
            ckpt_file = os.listdir(ckpt_path)[:-1]
            return os.path.join(ckpt_path, ckpt_file)
    
def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create your data generator
    data = TemplateGenerator(config)
    
    # create an instance of the model you want
    model = TemplateModel(config)

    # create trainer and pass all the previous components to it
    trainer = TemplateTrainer(model, data, config)
    
    # load model if exists
    if get_ckpt_path():
        ckpt_path = get_ckpt_path()
        model.load_params(ckpt_path)
    
    # here you train your model
    trainer.train()

if __name__ == '__main__':
    main()