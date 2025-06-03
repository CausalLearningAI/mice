import pandas as pd
import argparse
import os
import time

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('./src')

from data import PPCI
from utils import AIPW, accuracy

def get_parser():
    parser = argparse.ArgumentParser(description='ISTAnt')
    parser.add_argument('--sc', type=str, default="all", help='Splitting criteria')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--force', type=bool, default=False, help='Force training')
    parser.add_argument('--task', type=str, default="or", help='Task')
    parser.add_argument('--ref_preprocessed', type=str, default="original", help='preprocessed or original in reference')
    parser.add_argument('--tar_preprocessed', type=str, default="original", help='preprocessed or original in target')

    return parser

def main(args):
    print(f"Splitting criteria: {args.sc}")
    encoders = ["dino", "clip","clip_large","vit","vit_large"]
    methods = ["IRM", "ERM","vREx","DERM"]
    lrs = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    n_seeds = 5
    n_exps = len(encoders) * len(methods) * len(lrs) * n_seeds
    results = pd.DataFrame(columns=['method','encoder','lr','seed','ATE_ref', 'ATE_std_ref', 'ATE_p_value_ref', 'PPATE_ref', 'PPATE_std_ref', 'PPATE_p_value_ref', 'ATE_tar', 'ATE_std_tar', 'ATE_p_value_tar', 'PPATE_tar', 'PPATE_std_tar', 'PPATE_p_value_tar', 'acc_ref', 'bacc_ref', 'acc_tar', 'bacc_tar'])
    i = 0
    start = time.time()
    for encoder in encoders:
        reference = PPCI(encoder = encoder,
                    token = "class",
                    task = args.task,
                    split_criteria = args.sc,
                    environment = "supervised",
                    batch_size = 256,
                    num_proc = 2,
                    verbose = False,
                    data_dir = 'data/istant_lq',
                    results_dir = f'data/istant_lq/{args.ref_preprocessed}',
                    preprocessed = args.ref_preprocessed)
        target = PPCI(encoder = encoder,
                    token = "class",
                    task = args.task,
                    split_criteria = args.sc,
                    environment = "supervised",
                    batch_size = 256,
                    num_proc = 2,
                    verbose = False,
                    data_dir = 'data/istant_hq',
                    results_dir = f'data/istant_hq/{args.tar_preprocessed}',
                    preprocessed = args.tar_preprocessed)

        for method in methods:
            #for lr in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]:
            for lr in lrs:
                for seed in range(n_seeds):
                #for seed in range(1):
                    print(f"Encoder: {encoder}, Method: {method}, LR: {lr}, Seed: {seed}", flush=True)
                    if i>0:
                        partial = time.time() - start
                        print(f"Run {i}/{n_exps} ({int(partial // 3600)}h {int((partial % 3600) // 60)}m {int(partial % 60)}s/ {int((partial / i) * n_exps // 3600)}h {int(((partial / i) * n_exps % 3600) // 60)}m {int((partial / i) * n_exps % 60)}s)", flush=True)
                    reference.train(add_pred_env="supervised", 
                            hidden_layers = 2,
                            hidden_nodes = 256,
                            batch_size = 256,
                            lr = lr,
                            seed = seed,
                            num_epochs = args.num_epochs,
                            save = True,
                            verbose = False,
                            force = args.force,
                            method = method,
                            ic_weight = 10)
                    target.results_dir = f'data/istant_lq/{args.ref_preprocessed}'
                    target.train(add_pred_env="supervised", 
                                hidden_layers = 2,
                                hidden_nodes = 256,
                                batch_size = 256,
                                lr = lr,
                                seed = seed,
                                num_epochs = args.num_epochs,
                                save = False,
                                verbose = False,
                                method = method,
                                ic_weight = 10)
                    target.results_dir = f'data/istant_hq/{args.tar_preprocessed}'

                    print("Reference dataset:", flush=True)
                    acc_ref, bacc_ref = accuracy(reference)
                    print(f"Statistics: Accuracy={acc_ref:.3f}, Balanced Accuracy={bacc_ref:.3f}", flush=True)
                    ATE_ref, ATE_std_ref, ATE_p_value_ref = AIPW(reference, pred=False)
                    PPATE_ref, PPATE_std_ref, PPATE_p_value_ref = AIPW(reference, pred=True)
                    print(f"Causal: ATE={ATE_ref:.3f}±{ATE_std_ref:.3f}, PPATE={PPATE_ref:.3f}±{PPATE_std_ref:.3f}", flush=True)
                    
                    print("Target dataset:", flush=True)
                    acc_tar, bacc_tar = accuracy(target)
                    print(f"Statistics: Accuracy={acc_tar:.3f}, Balanced Accuracy={bacc_tar:.3f}", flush=True)
                    ATE_tar, ATE_std_tar, ATE_p_value_tar = AIPW(target, pred=False)
                    PPATE_tar, PPATE_std_tar, PPATE_p_value_tar = AIPW(target, pred=True)
                    print(f"Causal: ATE={ATE_tar:.3f}±{ATE_std_tar:.3f}, PPATE={PPATE_tar:.3f}±{PPATE_std_tar:.3f}", flush=True)

                    i += 1
                    results.loc[i] = [method, encoder, lr, seed, ATE_ref, ATE_std_ref, ATE_p_value_ref, PPATE_ref, PPATE_std_ref, PPATE_p_value_ref, ATE_tar, ATE_std_tar, ATE_p_value_tar, PPATE_tar, PPATE_std_tar, PPATE_p_value_tar, acc_ref, bacc_ref, acc_tar, bacc_tar]
        if not os.path.exists(f'results/ants/generalization'):
            os.makedirs(f'results/ants/generalization')
        pp = args.ref_preprocessed.replace("preprocessed/", "")+"->" + args.tar_preprocessed.replace("preprocessed/", "")
        results.to_csv(f"results/ants/generalization/pp_{pp}_task_{args.task}_sc_{args.sc}.csv", index=False)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)