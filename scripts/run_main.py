import argparse
import pandas as pd
import time

import os
import sys
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

from data import PPCI
from utils import get_time_string

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/v2", help="Path to the data directory")
    parser.add_argument("--results_dir", type=str, default="./results/v2", help="Path to the results directory")
    parser.add_argument("--hidden_nodes", type=int, default=256, help="Number of nodes per hidden layer")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--num_proc", type=int, default=2, help="Number of processes")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose")
    parser.add_argument("--generate", type=bool, default=False, help="Generate the dataset")
    parser.add_argument("--split_criteria", type=str, default="experiment", help="Splitting criteria for the dataset")
    parser.add_argument('--preprocessed', type=str, default="original", help='preprocessed or original')

    return parser

def main(args):
    methods = ["ERM", "IRM", "vREx", "DERM"]
    encoders = ["dino"]#, "clip_large", "clip", "mae", "vit", "vit_large"]
    tokens = ["class"]#, "mean", "all"]
    #split_criterias = ["experiment", "experiment_easy", "position", "position_easy", "random", "random_easy"]
    tasks = ["or","all"] #, "yellow", "blue", "sum", "all"]
    backgrounds = [True, False]
    #hidden_layerss = [1,2]
    cls_names = ["Transformer", "ConvNet", "MLP"]
    lrs = [0.05, 0.005, 0.0005]
    batch_sizes = [32, 64, 128]
    seeds = list(range(3))

    n_exp = len(methods)*len(encoders)*len(tokens)*len(tasks)*len(lrs)*len(seeds)*len(batch_sizes)* len(backgrounds)* len(cls_names)
    start_time = time.time()
    results = []
    #results = pd.DataFrame(columns=["method","encoder", "token", "split_criteria", "hidden_layers", "task", "lr", "seed", "color", "acc_train", "bacc_train", "acc_val", "bacc_val", "acc", "bacc", "ATE_train", "ATE_train_std", "PPATE_train", "PPATE_train_std", "ATE", "ATE_std", "PPATE", "PPATE_std"])
    k = 0
    for method in methods:
        for encoder in encoders:
            print("Encoder: ", encoder, flush=True)
            for token in tokens:
                print("Token: ", token, flush=True)
                for task in tasks:
                    print("Task: ", task, flush=True)
                    for background in backgrounds:
                        print("Background: ", background, flush=True)
                        for batch_size in batch_sizes:
                            print("Batch Size: ", batch_size, flush=True)
                            dataset = PPCI(encoder = encoder,
                                        token = token,
                                        task = task,
                                        split_criteria = args.split_criteria,
                                        environment = "supervised",
                                        batch_size = batch_size, 
                                        background = background,
                                        num_proc = args.num_proc,
                                        data_dir = args.data_dir,
                                        results_dir = args.results_dir,
                                        generate = args.generate,
                                        verbose = args.verbose)
                            for cls_name in cls_names:
                                print("Classifier: ", cls_name, flush=True)
                                for lr in lrs:
                                    print("Learning Rate: ", lr, flush=True)
                                    for seed in seeds: 
                                        print("Seed: ", seed, flush=True)
                                        k +=1
                                        start_time_i = time.time()
                                        dataset.train(batch_size = batch_size,
                                                    num_epochs = args.num_epochs,
                                                    lr = lr,
                                                    add_pred_env = "supervised",
                                                    seed = seed,
                                                    hidden_nodes = args.hidden_nodes, 
                                                    verbose = args.verbose,
                                                    method = method,
                                                    ic_weight = 10,
                                                    save = True, 
                                                    cls_name = cls_name)
                                        end_time_i = time.time()
                                        print(f"Experiment {k}/{n_exp} completed; Speed: {round(end_time_i-start_time_i, 1)}s/train, Total time elapsed {get_time_string(end_time_i - start_time)} (out of {get_time_string((end_time_i - start_time)/k*n_exp)}).", flush=True)
                                        
                                        if task == "all":
                                            for color in ["yellow", "blue"]:
                                                result = dataset.evaluate(color=color, verbose=False)
                                                result["method"] = method
                                                result["encoder"] = encoder
                                                result["token"] = token
                                                result["task"] = task
                                                result["lr"] = lr
                                                result["seed"] = seed
                                                result["color"] = color
                                                result["best_epoch"] = dataset.model.best_epoch
                                                result["background"] = background
                                                result["batch_size"] = batch_size
                                                result["classifier"] = cls_name
                                                results.append(result)
                                        else:
                                            result = dataset.evaluate(color=task, verbose=False)
                                            result["method"] = method
                                            result["encoder"] = encoder
                                            result["token"] = token
                                            result["task"] = task
                                            result["lr"] = lr
                                            result["seed"] = seed
                                            result["color"] = task
                                            result["best_epoch"] = dataset.model.best_epoch
                                            result["background"] = background
                                            result["batch_size"] = batch_size
                                            result["classifier"] = cls_name
                                            results.append(result)
        
    if not os.path.exists(f"{args.data_dir}/ID/{args.split_criteria}"):
        os.makedirs(f"{args.data_dir}/ID/{args.split_criteria}")
    results = pd.DataFrame(results)
    results.to_csv(f"{args.data_dir}/ID/{args.split_criteria}/results.csv", index=False)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    


