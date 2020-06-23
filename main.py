#! /usr/bin/env python3
import argparse
import yaml

import mc_dropout
import train


# Main parser
parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers(title="commands", help="", required=True, dest="command")

# Training parser
parser_train = subparsers.add_parser("train", description="Trains the model with provided "
                                                          "parameters and outputs weights")
parser_train.add_argument("-o", "--output", type=str, required=True, help="directory to save model to")
parser_train.add_argument("-c", "--config", type=str, help="config with training parameters")
parser_train.add_argument("-g", "--gpu", type=int, nargs='*', help="list of available GPUs")
parser_train.add_argument("-t", "--tensorboard", type=str, default="./tb", help="directory for tensorboard logs")

# Evaluation parser
parser_eval = subparsers.add_parser("eval", description="Evaluates provided model on validation set")
parser_eval.add_argument("-m", "--model", type=str, required=True, help="path to saved model")
parser_eval.add_argument("-c", "--config", type=str,  help="config with training parameters")

# MC dropout parser
parser_mc_dropout = subparsers.add_parser("mc-dropout", description="Plots uncertainty boundaries for "
                                                                    "predicted bounding boxes")
parser_mc_dropout.add_argument("-m", "--model", type=str, required=True, help="path to saved model")
parser_mc_dropout.add_argument("-c", "--config", type=str, required=True, help="config with training parameters")
parser_mc_dropout.add_argument("-s", "--saving_folder", type=str, default="pics", help="directory to save pictures to")


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
    params = load_yaml(args.config) if args.config else {}

    if args.command == "train":
        train.train(output_model_dir=args.output, tb_path=args.tensorboard, device_id=args.gpu, **params)
    elif args.command == "eval":
        loss, score = train.eval(model_path=args.model, **params)
        print(f'Validation loss: {loss:.4f}\nValidation mAP score: {score:.4f}')
    elif args.command == "mc-dropout":
        mc_processor = mc_dropout.MCProcessor(model=args.model, nuscenes_version=params['nuscenes_version'],
                                              data_path=params["data_path"], n_scenes=params['n_scenes'])
        mc_processor.visualise_monte_carlo(batch_size=1, sample_id=21, n_samples=10,
                                           saving_folder=args.saving_folder)
