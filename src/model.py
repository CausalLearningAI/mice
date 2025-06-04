from transformers import ViTImageProcessor, AutoImageProcessor, AutoProcessor, SiglipImageProcessor
from transformers import ViTForImageClassification, ResNetForImageClassification, AutoModel, CLIPVisionModel, ViTMAEModel, SiglipVisionModel
from datasets import Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

def get_model(encoder_name, device="cpu"):
    if encoder_name == "dino":
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    elif encoder_name == "vit":
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    elif encoder_name == "vit_large":
        processor = SiglipImageProcessor.from_pretrained('google/siglip-base-patch16-512')
        model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-512').to(device)
    elif encoder_name == "resnet":
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
    elif encoder_name == "clip":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")        
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    elif encoder_name == "clip_large":
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336').to(device)
    elif encoder_name == "mae":
        processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-large')
        model = ViTMAEModel.from_pretrained('facebook/vit-mae-large').to(device)
    else:
        raise ValueError(f"Encoder name: {encoder_name} is not implemented.")
    return processor, model

def get_embeddings(data, encoder_name, batch_size=100, num_proc=4, data_dir="./data", token="class", verbose=True):
    if token in ["class", "mean"]:
        data_emb_dir = os.path.join(data_dir, "embeddings", token, encoder_name)
        if os.path.exists(data_emb_dir):
            if len(os.listdir(data_emb_dir))>0:
                if verbose: print(f"Embeddings for '{data_dir}', encoder '{encoder_name}', token '{token}' already extracted.")
                embeddings = Dataset.load_from_disk(data_emb_dir)
                X = embeddings[encoder_name]
                X.encoder_name = encoder_name
                X.token = token
                return X
        else:
            os.makedirs(data_emb_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        processor, model = get_model(encoder_name, device)
        model.eval().requires_grad_(False)
        # data = data.map(lambda x: {"emb1": encoder(x["image"], model, processor, device)}, batch_size=batch_size, batched=True, num_proc=num_proc)
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_proc,
            pin_memory=True,
            shuffle=False,
        )
        embeddings = []
        for batch in tqdm(dataloader):
            embedding = encoder(batch["clip"], model, processor, device, token)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, 0)
        embeddings = Dataset.from_dict({encoder_name: embeddings.tolist()})
        embeddings.set_format(type="torch", columns=[encoder_name])
        embeddings.save_to_disk(data_emb_dir)
        if verbose: print(f"Embeddings from encoder '{encoder_name}' token '{token}' computed and saved correctly.")
        X = embeddings[encoder_name]
        X.encoder_name = encoder_name
        X.token = token
    elif token=="all":
        embeddings_class = get_embeddings(data, encoder_name, batch_size=batch_size, num_proc=num_proc, data_dir=data_dir, token="class", verbose=verbose)
        embeddings_mean = get_embeddings(data, encoder_name, batch_size=batch_size, num_proc=num_proc, data_dir=data_dir, token="mean", verbose=verbose)
        X = torch.cat((embeddings_class, embeddings_mean), dim=1)
        X.token = "all"
    else:
        raise ValueError("Token criteria not recognized. Please select between: 'class', 'mean', 'all'.")
    X.encoder_name = encoder_name
    return X

def encoder(x, model, processor, device, token="class"):

    B, T, C, H, W = x.shape
    x = x.to(device)
    x = x.view(B * T, C, H, W)

    inputs = processor(images=x, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    encoder_name_full = model.config._name_or_path

    if ("vit" in encoder_name_full) or ("dino" in encoder_name_full) or ("siglip" in encoder_name_full):
        if token == "class":
            emb = outputs.hidden_states[-1][:, 0]  # (B*T, D)
        elif token == "mean":
            emb = outputs.hidden_states[-1][:, 1:].mean(dim=1)  # (B*T, D)
        else:
            raise ValueError("Token criteria not recognized. Please select between: 'class', 'mean'.")
    elif ("resnet" in encoder_name_full):
        emb = outputs.hidden_states[-1].mean(dim=[2, 3])  # (B*T, D)
    else:
        raise ValueError(f"Unknown model class: {encoder_name_full}")

    return emb.view(B, T, -1).to("cpu")

def get_output_size(task):
    if task == "all":
        return 2  # Double Binary classification
    elif task == "sum":
        return 3  # Three classes [0, 1, 2]
    elif task in ["blue", "yellow", "or"]:
        return 1  # Single class (e.g., binary classification)
    else:
        raise ValueError(f"Task '{task}' is not recognized. Choose from 'all', 'sum', 'blue', 'yellow', or 'or'.")

def get_classifier(cls_name, task, emb_size=768, num_frames=7, hidden_nodes=128, kernel_size=3):  
    if cls_name=="Transformer":
        return ViVit(task=task,
                     emb_size=emb_size, 
                     num_frames=num_frames,
                     hidden_nodes=hidden_nodes, #128
                     )
    elif cls_name=="ConvNet":
        return TemporalConv(task=task, 
                            emb_size=emb_size, 
                            filters=hidden_nodes, #128
                            kernel_size=kernel_size,
                            )
    elif cls_name=="MLP":
        return MLP(task=task, 
                   emb_size=emb_size, 
                   hidden_nodes=hidden_nodes, #128
                   )

class ViVit(nn.Module):
    def __init__(self, task, emb_size=768, num_frames=7, hidden_nodes=128):
        super().__init__()
        self.task = task
        self.output_size = get_output_size(task)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_frames + 1, emb_size), requires_grad=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=4,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.head = nn.Sequential(
            nn.Linear(emb_size, hidden_nodes), # [B, D] --> [B, H]
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes//4), # [B, H] --> [B, H//4]
            nn.ReLU(),
            nn.Linear(hidden_nodes//4, self.output_size)  # Binary classification (logit output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                nn.init.xavier_uniform_(m.self_attn.in_proj_weight)
                if m.self_attn.in_proj_bias is not None:
                    nn.init.constant_(m.self_attn.in_proj_bias, 0.0)
                nn.init.xavier_uniform_(m.linear1.weight)
                if m.linear1.bias is not None:
                    nn.init.constant_(m.linear1.bias, 0.0)
                nn.init.xavier_uniform_(m.linear2.weight)
                if m.linear2.bias is not None:
                    nn.init.constant_(m.linear2.bias, 0.0)

    def forward(self, x):
        B, T, D = x.shape  # x: [B, T, D]
        cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        x = x + self.pos_embed[:, :T + 1]  # Add positional encoding [B, T+1, D]
        x = self.transformer(x)  # [B, T+1, D]
        return self.head(x[:, 0]) # [B, D] → [B, O] e.g. O=2 --> [-1.8, 0.4]
    
    def probs(self, X):
        if self.task=="sum":
            return self.forward(X).softmax(dim=-1) # [0.7, 0.1, 0.2]
        else:
            return self.forward(X).sigmoid() # [0.8, 0.4]
    def pred(self, X):
        if self.task=="sum":
            return torch.argmax(self.forward(X), dim=-1) # [0]
        else:
            return self.probs(X).round() # [1, 0]
    def cond_exp(self, X):
        if self.task=="sum":
            values = torch.tensor(range(3)).float().to(self.device)
            probs = self.probs(X)
            return torch.matmul(probs, values) # [0.5]
        else:
            return self.probs(X) # [0.8, 0.4]
        
class TemporalConv(nn.Module):
    def __init__(self, task, emb_size=768, filters=128, kernel_size=3):
        super().__init__()
        self.task = task
        self.output_size = get_output_size(task)

        self.encoder = nn.Sequential(
            nn.Conv1d(emb_size, filters, kernel_size=kernel_size, padding=0), # [B, D, T] --> [B, F, T-2]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # [B, F, T-2] --> [B, F, 1]
            nn.Flatten(), # [B, F, 1] --> [B, F]
        )

        self.head = nn.Sequential(
            nn.Linear(filters, filters//4), # [B, F] --> [B, F//4]
            nn.ReLU(),
            nn.Linear(filters//4, self.output_size)  # [B, F//4] --> [B, O] Binary classification (logit output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, T, D] --> [B, D, T]
        x = self.encoder(x)  # [B, F]
        return self.head(x) # [B, F] e.g. O=2 --> [-1.8, 0.4]
    
    def probs(self, X):
        if self.task=="sum":
            return self.forward(X).softmax(dim=-1) # [0.7, 0.1, 0.2]
        else:
            return self.forward(X).sigmoid() # [0.8, 0.4]
    def pred(self, X):
        if self.task=="sum":
            return torch.argmax(self.forward(X), dim=-1) # [0]
        else:
            return self.probs(X).round() # [1, 0]
    def cond_exp(self, X):
        if self.task=="sum":
            values = torch.tensor(range(3)).float().to(self.device)
            probs = self.probs(X)
            return torch.matmul(probs, values) # [0.5]
        else:
            return self.probs(X) # [0.8, 0.4]

class MLP(nn.Module):
    def __init__(self, task, emb_size=768, hidden_nodes=128):
        super().__init__()
        self.task = task
        self.output_size = get_output_size(task)

        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # [B, D, T] --> [B, D, 1]
            nn.Flatten(), # [B, D, 1] --> [B, D]
        )

        self.head = nn.Sequential(
            nn.Linear(emb_size, hidden_nodes), # [B, D] --> [B, H]
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes//4), # [B, H] --> [B, H//4]
            nn.ReLU(),
            nn.Linear(hidden_nodes//4, self.output_size)  # [B, H//4] --> [B, O] Binary classification (logit output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, T, D] --> [B, D, T]
        x = self.encoder(x)  # [B, D]
        return self.head(x) # [B, O] e.g. O=2 --> [-1.8, 0.4]
    
    def probs(self, X):
        if self.task=="sum":
            return self.forward(X).softmax(dim=-1) # [0.7, 0.1, 0.2]
        else:
            return self.forward(X).sigmoid() # [0.8, 0.4]
    def pred(self, X):
        if self.task=="sum":
            return torch.argmax(self.forward(X), dim=-1) # [0]
        else:
            return self.probs(X).round() # [1, 0]
    def cond_exp(self, X):
        if self.task=="sum":
            values = torch.tensor(range(3)).float().to(self.device)
            probs = self.probs(X)
            return torch.matmul(probs, values) # [0.5]
        else:
            return self.probs(X) # [0.8, 0.4]
