#! /usr/bin/env python3
import argparse
import yaml

import matplotlib.pyplot as plt

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
parser_train.add_argument("-d", "--data", type=str, default="./data", help="directory with nuScenes dataset")
parser_train.add_argument("-g", "--gpu", type=int, nargs='*', help="list of available GPUs")
parser_train.add_argument("-t", "--tensorboard", type=str, default="./tb", help="directory for tensorboard logs")

# Evaluation parser
parser_eval = subparsers.add_parser("eval", description="Evaluates provided model on validation set")
parser_eval.add_argument("-m", "--model", type=str, required=True, help="path to saved model")
parser_eval.add_argument("-c", "--config", type=str,  help="config with training parameters")
parser_eval.add_argument("-d", "--data", type=str, default="./data", help="directory with nuScenes dataset")

# MC dropout parser
parser_mc_dropout = subparsers.add_parser("mc-dropout", description="Plots uncertanty boundaries for ")
parser_mc_dropout.add_argument("-m", "--model", type=str, required=True, help="path to saved model")
parser_mc_dropout.add_argument("-c", "--config", type=str, required=True, help="config with training parameters")
parser_mc_dropout.add_argument("-d", "--data", type=str, default="./data", help="directory with nuScenes dataset")


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
        train.train(data_path=args.data, output_model_dir=args.output, tb_path=args.tensorboard,
                    device_id=args.gpu, **params)
    elif args.command == "eval":
        loss, score = train.eval(data_path=args.data, model_path=args.model, **params)
        print(f'Validation loss: {loss:.4f}\nValidation mAP score: {score:.4f}')
    elif args.command == "mc-dropout":
        mc_processor = mc_dropout.McProcessor(data_path=args.data, model_path=args.model,
                                              version=params['nuscenes_version'], n_scenes=params['n_scenes'])
        fig, ax_gt, ax_pred = mc_processor.visualise_montecarlo(batch_size=1, frame_id=21, n_samples=10,
                                                                save_imgs=False)
        plt.show()
