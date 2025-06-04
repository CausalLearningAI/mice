import torch
import random
import numpy as np
import pandas as pd
import time
import os

import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from scipy.stats import ttest_1samp

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_time_components(total_seconds):
    days = total_seconds // (24 * 3600)
    total_seconds %= (24 * 3600)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    total_seconds %= 60
    seconds = total_seconds
    return int(days), int(hours), int(minutes), int(seconds)

def get_time_string(total_seconds):
    days, hours, minutes, seconds = get_time_components(total_seconds)
    time_str = ''
    if days > 0:
        time_str += f'{days}d '
    if hours > 0:
        time_str += f'{hours}h'
    if minutes > 0:
        time_str += f'{minutes}m'
    time_str += f'{seconds}s'
    return time_str

def get_metric(Y, Y_hat, metric="accuracy"):
        if metric == "accuracy":
            metric =  (Y_hat == Y).float().mean()
        elif metric == "balanced_acc":
            TP = ((Y == 1) & (Y_hat == 1)).sum()
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            TN = ((Y != 1) & (Y_hat != 1)).sum()
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificy = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            metric = (recall+specificy)/2
        elif metric == "recall":
            TP = ((Y == 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            metric = TP / (TP + FN)
        elif metric == "precision":
            TP = ((Y == 1) & (Y_hat == 1)).sum()
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            metric = TP / (TP + FP)
        elif metric == "mse":
            metric =  ((Y_hat-Y)**2).mean()
        elif metric == "overestimate":
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            metric = (FP-FN)/len(Y)
        elif metric == "tr_equality":
            FP = ((Y != 1) & (Y_hat == 1)).sum()
            FN = ((Y == 1) & (Y_hat != 1)).sum()
            metric =  FN / FP
        else:
            raise ValueError(f"Metric '{metric}' not implented.")
        return metric.item()

def AIPW(dataset, treatment=2, control=1, pred=False, subset=None, color=None):
    settings = pd.read_csv(f'{dataset.data_dir}/experiments_settings.csv')#.dropna()
    settings = settings[settings["Valid"]==1]
    settings['Position'] = settings.apply(lambda x: x["Experiment"][1], axis=1).astype(int)
    settings['Experiment'] = settings.apply(lambda x: ord(x["Experiment"][0]) - ord('a'), axis=1).astype(int)
    if subset:
        split = get_split_settings(settings, split_criteria=dataset.split_criteria)
        if subset == "train":
            settings = settings[split]
        elif subset == "val":
            settings = settings[~split]
        else:
            raise ValueError(f"Subset '{subset}' not implemented.")

    settings = settings[settings["Treatment"].isin([treatment, control])]
    T = settings["Treatment"].replace({treatment: 1, control: 0})
    Y_ = dataset.supervised['Y_hat' if pred else "Y"]
    if dataset.task=="all":
        if color=="yellow":
            Y_ = Y_[:,0]
        elif color=="blue":
            Y_ = Y_[:,1]
        else:
            raise ValueError(f"Invalid color '{color}', please select between: 'blue', 'yellow'.")
    Y = settings.apply(lambda x: Y_[(dataset.supervised['source_data']["experiment"] == x["Experiment"]) & (dataset.supervised['source_data']["position"] == x["Position"])].sum().item(), axis=1)
    covariates = ["Position X", "Position Y", "Hour", "Date"]
    W = settings[covariates]
    if "Annotator" in settings.columns:
        W["Annotator"] = settings["Annotator"].astype('category').cat.codes
    W["Date"] = W["Date"].astype('category').cat.codes
    W = torch.tensor(W.values, dtype=torch.float32)
    T = torch.tensor(T.values, dtype=torch.float32).squeeze()
    Y = torch.tensor(Y.values, dtype=torch.float32).squeeze()

    ps = T.mean().item()
    N = len(T)
    model_outcome = XGBRegressor()
    model_outcome.fit(X = torch.cat((W, T.reshape(N, 1)), dim=1), y = Y)
    mu0 = model_outcome.predict(torch.cat((W, torch.zeros(N, 1)), dim=1)) #
    mu1 = model_outcome.predict(torch.cat((W, torch.ones(N, 1)), dim=1)) #Y[(T==1)[:,0]].mean().numpy()
    ite = mu1-mu0 + T.numpy() * (Y.numpy()-mu1) / ps - (1-T.numpy()) * (Y.numpy()-mu0) / (1-ps) 
    ATE = ite.mean()
    ATE_std = np.sqrt(ite.var()/N)
    p_value = ttest_1samp(ite, 0, alternative='greater')[1] #or smaller
    return ATE, ATE_std, p_value

def accuracy(dataset, subset=None, color="blue"):
    if dataset.task=="all":
        if color=="yellow":
            y = dataset.supervised["Y"][:,0]
            y_hat = dataset.supervised["Y_hat"][:,0]
        elif color=="blue":
            y = dataset.supervised["Y"][:,1]
            y_hat = dataset.supervised["Y_hat"][:,1]
        else:
            raise ValueError(f"Invalid color '{color}', please select between: 'blue', 'yellow'.")
    else:
        y = dataset.supervised["Y"]
        y_hat = dataset.supervised["Y_hat"]

    if not subset:
        y_hat = np.round(y_hat)
    elif subset == "train":
        y = y[dataset.supervised["split"]]
        y_hat = np.round(y_hat[dataset.supervised["split"]])
    elif subset == "val":
        y = y[~dataset.supervised["split"]]
        y_hat = np.round(y_hat[~dataset.supervised["split"]])
    else:
        raise ValueError(f"Subset '{subset}' not implemented.")
    
    bacc = get_metric(y, y_hat, metric="balanced_acc")
    acc = get_metric(y, y_hat, metric="accuracy")
    return acc, bacc

def get_split_settings(settings, split_criteria="random"):
    if split_criteria=="all":
        split = (settings["experiment"] >= 0) # tr_ration: 1
    elif split_criteria=="treatment0":
        split = (settings["Treatment"] == 0) # tr_ration: 1/3
    elif split_criteria=="treatment1":
        split = (settings["Treatment"] == 1) # tr_ration: 1/3
    elif split_criteria=="treatment2":
        split = (settings["Treatment"] == 2) # tr_ration: 1/3
    elif split_criteria=="experiment0" or split_criteria=="experiment":
        split = (settings["Experiment"] == 0) # tr_ration: 1/5
    elif split_criteria=="experiment1":
        split = (settings["Experiment"] == 1) # tr_ration: 1/5
    elif split_criteria=="experiment_easy":
        split = (settings["Experiment"] != 3) # tr_ration: 4/5 # previously 4
    elif split_criteria=="position":
        split = (settings["Position"] == 1) # tr_ration: 1/9
    elif split_criteria=="position_easy":
        split = (settings["Position"] != 8) # tr_ration: 8/9
    elif split_criteria=="random":
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        masks = [((settings['Experiment'] == e) & (settings['Position'] == p)) for e, p in zip(exps, poss)]
        split = pd.concat(masks, axis=1).any(axis=1)
    elif split_criteria=="random_easy":
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        masks = [((settings['Experiment'] == e) & (settings['Position'] == p)) for e, p in zip(exps, poss)]
        split = ~pd.concat(masks, axis=1).any(axis=1)
    else:
        raise ValueError(f"Split criteria {split_criteria} doesn't exist. Please select a valid splitting criteria: 'experiment', 'position' and 'random'.")
    return split

def get_time(start_time):
    """
    Print the elapsed time since start_time.
    """
    elapsed_time = time.time() - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= (24 * 3600)
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds", flush=True)
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    total_size = param_size + buffer_size  # in bytes
    return total_size / (1024 ** 2)  # convert to MB