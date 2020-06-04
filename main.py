#! /usr/bin/env python3
import argparse
import yaml

import train


# Main parser
parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers(title="commands", help="", required=True, dest="command")

# Training parser
parser_train = subparsers.add_parser("train", description="Trains the model with provided "
                                                          "parameters and outputs weights")
parser_train.add_argument("-o", "--output", type=str, required=True, help="directory to save model to")
parser_train.add_argument("-c", "--config", type=str, help="config with training parameters")
parser_train.add_argument("-d", "--data", type=str, default="./data", help="directory with nuScenes dataset")

# Evaluation parser
parser_eval = subparsers.add_parser("eval", description="Evaluates provided model on validation set")
parser_eval.add_argument("-m", "--model", type=str, required=True, help="directory to saved model")
parser_eval.add_argument("-c", "--config", type=str,  help="config with training parameters")
parser_eval.add_argument("-d", "--data", type=str, default="./data", help="directory with nuScenes dataset")


def load_yaml(filepath: str) -> dict:
    """
    Load yaml config
    :param filepath: Path to config
    :return: parsed yaml
    """
    with open(filepath, "r") as stream:
        result = yaml.safe_load(stream)
    return result


if __name__ == "__main__":
    args = parser.parse_args()

    if args.command == "train":
        params = load_yaml(args.config) if args.config else {}
        train.train(args.data, args.output, **params)
    elif args.command == "eval":
        params = load_yaml(args.config) if args.config else {}
        train.eval(args.data, args.model, **params)
