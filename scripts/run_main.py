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
    parser.add_argument("--data_dir", type=str, default="./data/istant_hq", help="Path to the data directory")
    parser.add_argument("--results_dir", type=str, default="./results/istant_hq", help="Path to the results directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_nodes", type=int, default=256, help="Number of nodes per hidden layer")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--num_proc", type=int, default=2, help="Number of processes")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose")
    parser.add_argument('--preprocessed', type=str, default="original", help='preprocessed or original')

    return parser

def main(args):
    methods = ["ERM", "IRM", "vREx", "DERM"]
    encoders = ["dino"]#, "clip_large", "clip", "mae", "vit", "vit_large"]
    tokens = ["class"]#, "mean", "all"]
    split_criterias = ["experiment", "experiment_easy", "position", "position_easy", "random", "random_easy"]
    tasks = ["or"] #, "yellow", "blue", "sum", "all"]
    
    hidden_layerss = [2]# [1,2]
    lrs = [0.05, 0.005, 0.0005]
    seeds = [0, 1, 2, 3, 4]

    n_exp = len(methods)*len(encoders)*len(tokens)*len(tasks)*len(split_criterias)*len(hidden_layerss)*len(lrs)*len(seeds)
    k = 0
    start_time = time.time()
    #results = pd.DataFrame(columns=["encoder", "token", "split_criteria", "hidden_layers", "task", "lr", "seed", "color", "loss_val", "acc_val", "bacc_val", "TEB_val", "acc", "bacc", "TEB", "TEB_bin", "EAD"])#, 'best_epoch'])
    results = pd.DataFrame(columns=["method","encoder", "token", "split_criteria", "hidden_layers", "task", "lr", "seed", "color", "acc_train", "bacc_train", "acc_val", "bacc_val", "acc", "bacc", "ATE_train", "ATE_train_std", "PPATE_train", "PPATE_train_std", "ATE", "ATE_std", "PPATE", "PPATE_std"])
    for method in methods:
        for encoder in encoders:
            print("Encoder: ", encoder)
            for token in tokens:
                print("Token: ", token)
                for split_criteria in split_criterias:
                    print("Split Criteria: ", split_criteria)
                    for task in tasks:
                        print("Task: ", task)
                        dataset = PPCI(encoder = encoder,
                                    token = token,
                                    task = task,
                                    split_criteria = split_criteria,
                                    environment = "supervised",
                                    batch_size = args.batch_size, 
                                    num_proc = args.num_proc,
                                    data_dir = args.data_dir,
                                    results_dir = f'{args.data_dir}/{args.preprocessed}',
                                    preprocessed = args.preprocessed,
                                    verbose = args.verbose)
                        for hidden_layers in hidden_layerss:
                            print("Hidden Layers: ", hidden_layers)
                            for lr in lrs:
                                print("Learning Rate: ", lr)
                                for seed in seeds: 
                                    print("Seed: ", seed)
                                    k +=1
                                    start_time_i = time.time()
                                    dataset.train(batch_size = args.batch_size,
                                                num_epochs = args.num_epochs,
                                                lr = lr,
                                                add_pred_env = "supervised",
                                                seed = seed,
                                                hidden_layers = hidden_layers,
                                                hidden_nodes = args.hidden_nodes, 
                                                verbose = args.verbose,
                                                method = method,
                                                ic_weight = 10,
                                                save = True)
                                    end_time_i = time.time()
                                    print(f"Experiment {k}/{n_exp} completed; Speed: {round(end_time_i-start_time_i, 1)}s/train, Total time elapsed {get_time_string(end_time_i - start_time)} (out of {get_time_string((end_time_i - start_time)/k*n_exp)}).")
                                    
                                    if task == "all":
                                        for color in ["yellow", "blue"]:
                                            result = dataset.evaluate(color=color, verbose=False)
                                            result["method"] = method
                                            result["encoder"] = encoder
                                            result["token"] = token
                                            result["split_criteria"] = split_criteria
                                            result["hidden_layers"] = hidden_layers
                                            result["task"] = task
                                            result["lr"] = lr
                                            result["seed"] = seed
                                            result["color"] = color
                                            #result["best_epoch"] = dataset.model.best_epoch
                                            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
                                    else:
                                        result = dataset.evaluate(color=task, verbose=False)
                                        result["method"] = method
                                        result["encoder"] = encoder
                                        result["token"] = token
                                        result["split_criteria"] = split_criteria
                                        result["hidden_layers"] = hidden_layers
                                        result["task"] = task
                                        result["lr"] = lr
                                        result["seed"] = seed
                                        result["color"] = task
                                        #result["best_epoch"] = dataset.model.best_epoch
                                        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
        
    if not os.path.exists(f"results/ants/main/{args.data_dir[-2:]}"):
        os.makedirs(f"results/ants/main/{args.data_dir[-2:]}")
    results.to_csv(f"results/ants/main/{args.data_dir[-2:]}/pp_{args.preprocessed.replace('preprocessed/', '')}.csv")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    


