import argparse
import sys
sys.path.append('./src')
import torch
from utils import *
from trainer import train, evaluate
from summary_model import MultitaskDeletionAbsSummaryModel
from torch.nn import BCELoss, BCEWithLogitsLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(cli_args):
    args = init_setting(cli_args)
    model = MultitaskDeletionAbsSummaryModel(args).to(device)
    criterion = BCEWithLogitsLoss()
    # if args.do_eval:
    #     evaluate(args, model, criterion)

    if args.do_train:
        train(args, model, criterion)


    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_file", type=str, default='kobart.json')
    cli_parser.add_argument("--output_dir", type=str, default='kobart_ckpt')
    cli_args = cli_parser.parse_args()

    main(cli_args)
