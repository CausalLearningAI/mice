import torch
import pandas as pd
import numpy as np
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os
from datasets import Dataset

from model import get_embeddings, get_classifier
from train import train_, train_md
from visualize import plot_outcome_distribution
from utils import set_seed, check_folder, accuracy, AIPW

class PPCI():
    def __init__(self, 
                task="all", 
                encoder="dino", 
                token="class", 
                split_criteria="experiment", 
                reduce_fps_factor=15, 
                downscale_factor=1, 
                context=3,
                stride=2,
                batch_size=100, 
                num_proc=4, 
                environment="all", 
                generate=False, 
                data_dir="./data/v2", 
                results_dir="./results/v2", 
                background=False,
                verbose=False):
        
        # TODO: check generate option
        if environment in ["all", "supervised"]:
            self.supervised = load_env("supervised", 
                                    task=task, 
                                    encoder=encoder, 
                                    token=token,
                                    split_criteria=split_criteria,
                                    reduce_fps_factor=reduce_fps_factor, 
                                    downscale_factor=downscale_factor,
                                    batch_size=batch_size, 
                                    num_proc=num_proc,
                                    generate=generate,
                                    context=context,
                                    stride=stride,
                                    data_dir=data_dir,
                                    background=background,
                                    verbose=verbose)
            self.n_supervised = self.supervised["T"].shape[0]
        if environment in ["all", "unsupervised"]:
            self.unsupervised = load_env("unsupervised", 
                                    task=task, 
                                    encoder=encoder, 
                                    token=token,  
                                    split_criteria=split_criteria,
                                    reduce_fps_factor=reduce_fps_factor, 
                                    downscale_factor=downscale_factor,
                                    batch_size=batch_size, 
                                    num_proc=num_proc,
                                    generate=generate,
                                    context=context,
                                    stride=stride,
                                    data_dir=data_dir,
                                    background=background,
                                    verbose=verbose)
            self.n_unsupervised = self.unsupervised["T"].shape[0]
        self.task = task
        self.encoder = encoder
        self.token = token
        self.split_criteria = split_criteria
        self.data_dir = data_dir
        self.results_dir = results_dir
        if verbose: print("Prediction-Powered Causal Inference dataset successfully loaded.")
    
    def train(self, batch_size=256, num_epochs=10, lr=0.001, hidden_nodes=128, verbose=True, add_pred_env="supervised", seed=0, save=False, force=False, method="ERM", ic_weight=10, gpu=True, cfl=0, cls_name="Transformer"):
        set_seed(seed)
        if method=='DERM' and self.task=="all":
            raise ValueError("DERM method is not available (yet) for task 'all'.")
        if not method in ["ERM", "vREx", "DERM", "IRM"]:
            raise ValueError(f"Method '{method}' not defined. Please select between: 'ERM', 'vREx', 'DERM'.")
        method_ = method if "ERM" in method else method+"_"+str(ic_weight)
        model_path = os.path.join(self.results_dir, "models", self.encoder, cls_name, self.token, self.split_criteria, self.task, str(lr), str(seed), f"{method_}.pth")
        if os.path.exists(model_path) and not force:
            if verbose: print("Model already trained.")
            emb_size, num_frames = self.supervised["X"].shape[1], self.supervised["X"].shape[2]
            self.model = get_classifier(cls_name, self.supervised["Y"].task, emb_size=emb_size, num_frames=num_frames, hidden_nodes=hidden_nodes, kernel_size=3)
            self.model.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
            self.model.load_state_dict(torch.load(model_path, map_location=self.model.device, weights_only=True))
            self.model.to(self.model.device)
        else:
            if method in ["vREx", "IRM"]:
                self.model = train_md(self.supervised, 
                                    batch_size=batch_size, 
                                    num_epochs=num_epochs, 
                                    lr=lr, 
                                    hidden_nodes = hidden_nodes, 
                                    verbose=verbose,
                                    ic_weight=ic_weight,
                                    method=method,
                                    gpu=gpu,
                                    cfl=cfl,
                                    cls_name=cls_name)
            else:
                self.model = train_(self.supervised, 
                                    batch_size=batch_size, 
                                    num_epochs=num_epochs, 
                                    lr=lr, 
                                    hidden_nodes = hidden_nodes, 
                                    verbose=verbose,
                                    decondounded = method=="DERM",
                                    gpu=gpu,
                                    cfl=cfl,
                                    cls_name=cls_name)
            if save:
                method_ = method if "ERM" in method else method+"_"+str(ic_weight)
                model_dir = os.path.join(self.results_dir, "models", cls_name, self.encoder, self.token, self.split_criteria, self.task, str(lr), str(seed))
                check_folder(model_dir)
                torch.save(self.model.state_dict(), os.path.join(model_dir, f"{method_}.pth"))
        if add_pred_env in ["supervised", "unsupervised"]:
            self.add_pred(add_pred_env)
        elif add_pred_env=="all":
            self.add_pred("supervised")
            self.add_pred("unsupervised")
        else:
            raise ValueError(f"Invalid add_pred_env argument '{add_pred_env}', please select among: 'supervised', 'unsupervised', or 'all'.")
    
    def plot_out_distribution(self, save=True, total=True):
        # TODO: check if works/ update
        if self.task=="all":
            plot_outcome_distribution(self.supervised, save=save, total=total, results_dir=self.results_dir)
        else:
            raise ValueError("Plot available only for task: 'all'.")

    def add_pred(self, environment="supervised", max_batch_size=1000):
        if hasattr(self, 'model'):
            device = self.model.device
            with torch.no_grad():
                if environment=="supervised":
                    self.supervised["Y_hat"] = []
                    for i in range(0, self.supervised["X"].shape[0], max_batch_size):
                        batch_X = self.supervised["X"][i:i+max_batch_size].to(device)
                        batch_Y_hat = self.model.cond_exp(batch_X).to("cpu").squeeze()
                        self.supervised["Y_hat"].append(batch_Y_hat)
                    self.supervised["Y_hat"] = torch.cat(self.supervised["Y_hat"], dim=0)
                elif environment=="unsupervised":
                    self.unsupervised["Y_hat"] = []
                    for i in range(0, self.unsupervised["X"].shape[0], max_batch_size):
                        batch_X = self.unsupervised["X"][i:i+max_batch_size].to(device)
                        batch_Y_hat = self.model.cond_exp(batch_X).to("cpu").squeeze()
                        self.unsupervised["Y_hat"].append(batch_Y_hat)
                    self.unsupervised["Y_hat"] = torch.cat(self.unsupervised["Y_hat"], dim=0)
                else:
                    raise ValueError(f"Environment '{environment}' not defined.")
        else:
            raise ValueError("Train the model first, before computing the inference step.")
    
    def evaluate(self, color="blue", T_control=1, T_treatment=2, verbose=False):
        if "Y_hat" in self.supervised:
            # stats
            acc_train, bacc_train = accuracy(self, subset="train", color=color)
            acc_val, bacc_val = accuracy(self, subset="val", color=color)
            acc, bacc = accuracy(self, subset=None, color=color)
            # causal
            ATE_train, ATE_train_std, _ = AIPW(self, treatment=T_treatment, control=T_control, pred=False, color=color, subset="train")
            PPATE_train, PPATE_train_std, _ = AIPW(self, treatment=T_treatment, control=T_control, pred=True, color=color, subset="train")
            ATE, ATE_std, _ = AIPW(self, treatment=T_treatment, control=T_control, pred=False, color=color, subset=None)
            PPATE, PPATE_std, _ = AIPW(self, treatment=T_treatment, control=T_control, pred=True, color=color, subset=None)
      
            metric = {
                "acc_train": acc_train,
                "bacc_train": bacc_train,
                "acc_val": acc_val,
                "bacc_val": bacc_val,
                "acc": acc,
                "bacc": bacc,
                "ATE_train": ATE_train,
                "ATE_train_std": ATE_train_std,
                "PPATE_train": PPATE_train,
                "PPATE_train_std": PPATE_train_std,
                "ATE": ATE,
                "ATE_std": ATE_std,
                "PPATE": PPATE,
                "PPATE_std": PPATE_std,
            }
            # TODO: add structured printing
            if verbose: print(metric)
            return metric
        else:
            raise ValueError("Train the model and predict the labels on the supervised dataset before measuring the performances.")
    
    def get_examples(self, n, environment="supervised", validation=False):
        if environment=="supervised":
            if validation:
                val_indeces = torch.nonzero(~self.supervised["split"]).squeeze()
                idxs = random.sample(val_indeces.tolist(), n)
            else:
                train_indeces = torch.nonzero(self.supervised["split"]).squeeze()
                idxs = random.sample(train_indeces.tolist(), n)
            exps = self.supervised["source_data"][idxs]["experiment"]
            poss = self.supervised["source_data"][idxs]["position"]
            exp = [chr(97+exp)+str(pos.item()) for exp, pos in zip(exps, poss)]
            frame = self.supervised["source_data"][idxs]["frame"]
            image = self.supervised["source_data"][idxs]["image"]
            Y = self.supervised["Y"][idxs] 
            if "Y_hat" in self.supervised:
                Y_hat = self.supervised["Y_hat"][idxs] 
            else:
                Y_hat = None
        elif environment=="unsupervised":
            idxs = torch.randint(0, self.n_unsupervised, (n,))
            image = self.unsupervised["source_data"][idxs]["image"]
            exps = self.supervised["source_data"][idxs]["experiment"]
            poss = self.supervised["source_data"][idxs]["position"]
            exp = [chr(97+exp)+str(pos.item()) for exp, pos in zip(exps, poss)]
            frame = self.unsupervised["source_data"][idxs]["frame"]
            Y = None
            if "Y_hat" in self.unsupervised:
                Y_hat = self.unsupervised["Y_hat"][idxs] 
            else:
                Y_hat = None
        else:
            raise ValueError(f"Environemnt '{environment}' not defined, please select between: 'supervised' and 'unsupervised'.")
        return image, Y, Y_hat, exp, frame

    def visualize_frame(self, save=True, k=6, detailed=True):
        train, test = False, False
        if hasattr(self, 'supervised'):
            if ("Y_hat" in self.supervised):
                train = True
        if hasattr(self, 'unsupervised'):
            if ("Y_hat" in self.unsupervised):
                test = True
        if train+test==0:
            raise ValueError("Generate first at least an environment, train a model and predict the corresponding labels before visualizing.")
        fig = plt.figure(figsize=(k*2.5, 0.4+4.4*train+2.2*test))
        ax = []
        if train: 
            # train
            clips, Ys, Y_hats, exps, frames = self.get_examples(k, environment="supervised", validation=False)
            for i, (clip, y, y_pred, exp, frame) in enumerate(zip(clips, Ys, Y_hats.round(), exps, frames)):
                img = clip[clip.shape[0] // 2]
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                y = [int(elem.item()) for elem in y.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + 1))
                if detailed: title = f"H: {y}, ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"H: {y}, ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[0].annotate('Training', xy=(0, 0.5), xytext=(-ax[0].yaxis.labelpad - 5, 0),
                            xycoords=ax[0].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
            # validation
            clips, Ys, Y_hats, exps, frames = self.get_examples(k, environment="supervised", validation=True)
            for i, (clip, y, y_pred, exp, frame) in enumerate(zip(clips, Ys, Y_hats.round(), exps, frames)):
                img = clip[clip.shape[0] // 2]
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                y = [int(elem.item()) for elem in y.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + k + 1))
                if detailed: title = f"H: {y}, ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"H: {y}, ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[k].annotate('Validation', xy=(0, 0.5), xytext=(-ax[k].yaxis.labelpad - 5, 0),
                            xycoords=ax[k].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
        if test:
            # test
            clips, _, Y_hats = self.get_examples(k, environment="unsupervised")
            for i, (clip, y_pred, exp, frame) in enumerate(zip(clips, Y_hats.round(), exps, frames)):
                img = clip[clip.shape[0] // 2]
                y_pred = [int(elem.item()) for elem in y_pred.unsqueeze(-1)]
                plt.rc('font', size=8)
                ax.append(fig.add_subplot(2*train+test, k, i + 2*train*k +1))
                if detailed: title = f"ML: {y_pred}\nExp: {exp}, Frame: {frame}"
                else: title = f"ML: {y_pred}"
                ax[-1].set_title(title)
                plt.imshow(img.permute(1, 2, 0))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
            ax[2*train*k].annotate('Test', xy=(0, 0.5), xytext=(-ax[2*train*k].yaxis.labelpad - 5, 0),
                            xycoords=ax[2*train*k].yaxis.label, textcoords='offset points',
                            fontsize=14, ha='center', va='center', rotation=90)
        if save: 
            results_example_dir = os.path.join(self.results_dir, "example_pred")
            if not os.path.exists(results_example_dir):
                os.makedirs(results_example_dir)
            title = f"{self.encoder}_{self.token}_task_{self.task}.png"
            path_fig = os.path.join(results_example_dir, title)
            plt.savefig(path_fig, bbox_inches='tight')
        else:
            plt.show()

    def __str__(self):
        return "Prediction-Powered Causal Inference dataset (PPCI object)"

    def __repr__(self):
        return "Prediction-Powered Causal Inference dataset (PPCI object)"

def get_outcome(dataset, task="all"):
    if task=="all":
        y = dataset["outcome"]
    elif task.lower()=="yellow":
        y = dataset["outcome"][:,0]
    elif task.lower()=="blue":
        y = dataset["outcome"][:,1]
    elif task.lower()=="sum":
        y = dataset["outcome"].sum(axis=1)
    elif task.lower()=="or":
        y = torch.logical_or(dataset["outcome"][:,0], dataset["outcome"][:,1]).float()
    else:
        raise ValueError(f"Task {task} not defined. Please select between: 'all', 'yellow', 'blue', 'sum', 'or'.")
    y.task = task
    return y

def get_split(dataset, split_criteria="random"):
    if split_criteria=="all":
        split = (dataset["experiment"] >= 0) # tr_ration: 1
    elif split_criteria=="treatment0":
        split = (dataset["treatment"] == 0) # tr_ration: 1/3
    elif split_criteria=="treatment1":
        split = (dataset["treatment"] == 1) # tr_ration: 1/3
    elif split_criteria=="treatment2":
        split = (dataset["treatment"] == 2) # tr_ration: 1/3
    elif split_criteria=="experiment0" or split_criteria=="experiment":
        split = (dataset["experiment"] == 0) # tr_ration: 1/5
    elif split_criteria=="experiment1":
        split = (dataset["experiment"] == 1) # tr_ration: 1/5
    elif split_criteria=="experiment_easy":
        split = (dataset["experiment"] != 3) # tr_ration: 4/5 # previously 4
    elif split_criteria=="position":
        split = (dataset["position"] == 1) # tr_ration: 1/9
    elif split_criteria=="position_easy":
        split = (dataset["position"] != 8) # tr_ration: 8/9
    elif split_criteria=="random":
        split = torch.zeros_like(dataset["experiment"], dtype=torch.bool)
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        for exp_i, pos_i in zip(exps,poss):
            split_i = (dataset["experiment"]==exp_i) & (dataset["position"]==pos_i)
            split = split | split_i # tr_ration: 1/5
    elif split_criteria=="random_easy":
        split = torch.ones_like(dataset["experiment"], dtype=torch.bool)
        exps = [0,0,1,1,2,2,3,3,4]
        poss = [2,3,4,5,1,2,3,4,9]
        for exp_i, pos_i in zip(exps,poss):
            split_i = (dataset["experiment"]!=exp_i) | (dataset["position"]!=pos_i)
            split = split & split_i # tr_ration: 4/5
    else:
        raise ValueError(f"Split criteria {split_criteria} doesn't exist. Please select a valid splitting criteria: 'experiment', 'position' and 'random'.")
    split.criteria = split_criteria
    return split

def get_covariates(dataset):
    covariates = ['pos_x', 'pos_y', 'exp_minute', 'experiment']
    W = torch.stack([dataset[covariate] for covariate in covariates[:-1]], dim=1)
    W_exp = torch.nn.functional.one_hot(dataset["experiment"], num_classes=len(dataset["experiment"].unique()))
    W = torch.cat([W, W_exp], dim=1)
    return W

def get_tracking(dataset):
    tracking = dataset["tracking"]
    if tracking is None:
        raise ValueError("Tracking data not available in the dataset.")
        # TODO: add script to generate tracking data if not available
    if isinstance(tracking, torch.Tensor):
        return tracking
    else:
        raise ValueError("Tracking data should be a PyTorch tensor.")

def load_env(environment='supervised', task="all", encoder="dino", token="class", split_criteria="experiment", generate=False, reduce_fps_factor=10, downscale_factor=1, batch_size=100, num_proc=4, data_dir="./data", background=False, verbose=False, context=3, stride=2):
    data_env_dir = os.path.join(data_dir, environment, "background" if background else "nobackground")
    if not os.path.exists(data_env_dir):
        os.makedirs(data_env_dir)
        generate = True
    if not os.path.exists(os.path.join(data_env_dir, "state.json")):
        generate = True
    if generate:           
        dataset = Dataset.from_generator(generator, 
                                         gen_kwargs={"reduce_fps_factor": reduce_fps_factor, 
                                                     "downscale_factor": downscale_factor, 
                                                     "environment":environment, 
                                                     "background":background, 
                                                     "data_dir":data_dir,
                                                     "context":context,
                                                     "stride":stride},
                                         num_proc=num_proc,)
        dataset.save_to_disk(data_env_dir)
        if verbose: print("Data generated and saved correctly.")
    else:
        dataset = Dataset.load_from_disk(data_env_dir)
    dataset.set_format(type="torch", columns=["clip", "treatment", "outcome", 'pos_x', 'pos_y', 'exp_minute', 'day_hour', 'frame', "experiment", "position", "tracking"], output_all_columns=True)
    dataset.environment = environment
    W = get_covariates(dataset)
    if ('v1' in data_dir) or ('v2' in data_dir):
        exp_id = W[:, -5:] @ np.array([0,1,2,3,4])
        pos_id = W[:, 0] + 1 + 3*(W[:, 1] + 1)
        E = (9*exp_id + pos_id).to(torch.int64)
    else:
        raise ValueError(f"Unknown 'environment' definition for dataset: {data_env_dir}")
    X = torch.cat([get_embeddings(dataset, encoder, batch_size=batch_size, num_proc=num_proc, data_dir=data_env_dir, token=token, verbose=verbose), 
                   get_tracking(dataset)], 
                   dim=-1)
    X.token = token
    X.encoder_name = encoder
    dataset_dict = {
        "source_data": dataset,
        "X": X,
        "Y": get_outcome(dataset, task=task),
        "split": get_split(dataset, split_criteria=split_criteria),
        "W": W, 
        "E": E,
        "T": dataset["treatment"],
    }
    if verbose: 
        print("Training Environments: ", np.unique(E[dataset_dict["split"]]))
        print("Validation Environments: ", np.unique(E[~dataset_dict["split"]]))
    return dataset_dict

def generator(reduce_fps_factor, downscale_factor, environment='supervised', background=False, data_dir="./data", context=3, stride=2):
    if environment == 'supervised':
        start_frame_column = 'Starting Frame'
        end_frame_column = 'End Frame Annotation'
    elif environment == 'unsupervised':
        start_frame_column = 'End Frame Annotation'
        end_frame_column = 'Valid until frame'
    else:
        raise ValueError(f'Unknown environment: {environment}')
    settings = pd.read_csv(f'{data_dir}/experiments_settings.csv')
    for id_exp, exp in enumerate(["a", "b", "c", "d", "e"]):
        print(f"Loading experiment {exp}")
        for pos in range(1, 10):
            print(f"Loading position {pos}")
            valid = int(settings[settings.Experiment == f'{exp}{pos}']["Valid"].values[0])
            if valid == 0:
                continue
            start_frame = int(settings[settings.Experiment == f'{exp}{pos}'][start_frame_column].values[0])
            end_frame = int(settings[settings.Experiment == f'{exp}{pos}'][end_frame_column].values[0])
            if end_frame-start_frame<1:
                continue
            treatment = settings[settings.Experiment == f'{exp}{pos}']['Treatment'].values[0].astype(int)
            fps = settings[settings.Experiment == f'{exp}{pos}']["FPS"].values[0].astype(int)/reduce_fps_factor
            day_hour = settings[settings.Experiment == f'{exp}{pos}']["Hour"].values[0]
            pos_x = settings[settings.Experiment == f'{exp}{pos}']["Position X"].values[0]
            pos_y = settings[settings.Experiment == f'{exp}{pos}']["Position Y"].values[0]

            # load clips
            clips = load_clips(exp, pos,
                               reduce_fps_factor=reduce_fps_factor, 
                               downscale_factor=downscale_factor, 
                               start_frame=start_frame, 
                               end_frame=end_frame,
                               data_dir=data_dir,
                               background=background,
                               context=context,
                               stride=stride)
            print(f"Clips shape: {clips.shape}", flush=True)
            # load annotations
            labels = load_labels(exp, pos, 
                                 reduce_fps_factor=reduce_fps_factor,
                                 start_frame=int(start_frame/reduce_fps_factor),
                                 end_frame=int(end_frame/reduce_fps_factor),
                                 data_dir=data_dir)
            print(f"Labels shape: {labels.shape}", flush=True)
            # load tracking data
            tracking = load_tracking(exp, pos,
                                    reduce_fps_factor=reduce_fps_factor, 
                                    start_frame=start_frame, 
                                    end_frame=end_frame,
                                    context=context,
                                    stride=stride,
                                    data_dir=data_dir)
            print(f"Tracking shape: {tracking.shape}", flush=True)
            for i in range(int((end_frame-start_frame)/reduce_fps_factor)):
                yield {
                    "experiment": id_exp,
                    'position': pos,                         
                    "pos_x": pos_x, # covariate                          
                    "pos_y": pos_y, # covariate   
                    "frame": i,
                    "clip": clips[i],
                    "treatment": treatment,
                    "outcome": labels[i,:],
                    "exp_minute": ((start_frame+i)/fps)//60, # covariate   
                    "day_hour": day_hour, # covariate   
                    "tracking": tracking[i], 
                }

def map_behaviour_to_label(behaviour):
    if behaviour == ' groom-yellow' or behaviour == ' groom-orange':
        return (1,0)
    if behaviour == ' groom-blue':
        return (0,1)
    if behaviour == ' groom-yellowandblue' or behaviour == ' groom-orangeandblue':
        return (1,1)
    else:
        raise ValueError(f'Unknown behaviour: {behaviour}')

def label_frame(frame_id, behaviors):
    yellow = 0
    blue = 0
    for _, row in behaviors.iterrows():
        if frame_id >= row[' Beginning-frame'] and frame_id < row[' End-frame']:
            yellow_i, blue_i = map_behaviour_to_label(row[' Behavior'])
            yellow += yellow_i
            blue += blue_i
    return (yellow,blue)
        
def load_labels(exp, pos, reduce_fps_factor, start_frame, end_frame, data_dir):
    behaviors_path = os.path.join(data_dir, f"behavior/{exp}{pos}.csv")
    behaviors = pd.read_csv(behaviors_path, skiprows=3, skipfooter=1, engine='python')
    if behaviors.shape[0]==0:
        return torch.zeros(end_frame-start_frame, 2, dtype=torch.float32)
    else:
        labels = [] # list of couples
        for i in range(start_frame, end_frame):
            labels.append(label_frame(i*reduce_fps_factor, behaviors))
        return torch.tensor(labels, dtype=torch.float32) # tensor Nx2
    
# def load_tracking(exp, pos, reduce_fps_factor, start_frame, end_frame, data_dir="./data"):
#     tracking_path = os.path.join(data_dir, f"tracking/position/{exp}{pos}.csv")
#     tracking = pd.read_csv(tracking_path, engine='python', index_col=0)
#     tracking_filtered = tracking.iloc[start_frame:end_frame:reduce_fps_factor, :]
#     return torch.tensor(tracking_filtered.values, dtype=torch.float32) # tensor Nx16

def load_tracking(exp, pos, reduce_fps_factor, start_frame, end_frame, context=3, stride=2, data_dir="./data"):
    tracking_path = os.path.join(data_dir, f"tracking/position/{exp}{pos}.csv")
    tracking = pd.read_csv(tracking_path, engine='python', index_col=0)
    tracking = tracking.iloc[start_frame:end_frame].reset_index(drop=True)

    num_frames, num_features = tracking.shape
    window_size = 2 * context + 1
    windows = []

    center_indices = list(range(0, num_frames, reduce_fps_factor))

    for center in center_indices:
        start_idx = center - context
        end_idx = center + context + 1
        pad_left = max(0, -start_idx)
        pad_right = max(0, end_idx - num_frames)
        start_idx_clamped = max(0, start_idx)
        end_idx_clamped = min(num_frames, end_idx)
        window = tracking.iloc[start_idx_clamped:end_idx_clamped].values
        if pad_left > 0:
            pad_vals = np.repeat(window[0:1], pad_left, axis=0)
            window = np.vstack((pad_vals, window))
        if pad_right > 0:
            pad_vals = np.repeat(window[-1:], pad_right, axis=0)
            window = np.vstack((window, pad_vals))
        windows.append(window)

    tracking_tensor = torch.tensor(np.stack(windows), dtype=torch.float32) # [N, 2*context+1, D]
    return tracking_tensor

def load_frames(exp, pos, reduce_fps_factor, downscale_factor, start_frame, end_frame, data_dir="./data", background=False):
    if background:
        video_name = f'background/focal/{exp}{pos}.mp4'
    else:
        video_name = f'nobackground/focal/{exp}{pos}.mp4'
    video_path = os.path.join(data_dir, "tracking", video_name)
    cap = cv2.VideoCapture(video_path)
    #original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Frame rate reduction
        if frame_count % reduce_fps_factor == 0:
            # Downscaling
            if downscale_factor<1:
                resized_frame = cv2.resize(frame, (0, 0), fx=downscale_factor, fy=downscale_factor)
            else: 
                resized_frame = frame
            # Convert to PyTorch tensor (RGB)
            tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1)[[2, 1, 0], :, :]
            frames.append(tensor_frame)
        frame_count += 1

    cap.release()
    return frames[start_frame:end_frame]

def load_clips(exp, pos, reduce_fps_factor, downscale_factor, start_frame, end_frame,
               data_dir="./data", background=False, context=0, stride=1):
    if background:
        video_name = f'background/focal/{exp}{pos}.mp4'
    else:
        video_name = f'nobackground/focal/{exp}{pos}.mp4'
    video_path = os.path.join(data_dir, "tracking", video_name)

    cap = cv2.VideoCapture(video_path)
    total_high_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    central_indices = list(range(start_frame, end_frame, reduce_fps_factor))
    clips = []
    for center_idx in central_indices:
        clip_indices = list(range(center_idx - context*stride, center_idx + context*stride + 1, stride))
        clip_indices = [max(0, min(i, total_high_fps - 1)) for i in clip_indices]

        clip_frames = []
        for idx in clip_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from video.")

            if downscale_factor < 1:
                frame = cv2.resize(frame, (0, 0), fx=downscale_factor, fy=downscale_factor)
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1)[[2, 1, 0], :, :]  # BGR to RGB
            clip_frames.append(tensor_frame)

        clips.append(torch.stack(clip_frames))  # (2*context+1, C, H, W)

    cap.release()
    return torch.stack(clips)  # (N, 2*context+1, C, H, W)