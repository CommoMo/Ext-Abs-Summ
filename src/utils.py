from attrdict import AttrDict
import json
import logging, os

LOG_PATH = './logs'

def get_logger(args):
        ### Set Loggers ###
    logger = logging.getLogger('[System]')
    return logger


def init_setting(cli_args):
    # read from config file and make args
    with open(os.path.join('configs', cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
        args['output_dir'] = cli_args.output_dir

    with open(os.path.join('configs', cli_args.config_file), 'w') as f:
        json.dump(args, f, indent='\t')

    ### Set Loggers ###
    logger = get_logger(args)
    logger_path = os.path.join(LOG_PATH, f'{args.model_type}.log')
    file_handler = logging.FileHandler(logger_path)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"  >> Current path: {os.getcwd()}")
    logger.info("Training/evaluation parameters {}".format(args))

    return args