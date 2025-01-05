from model import *
from utils import printm, init_weights, count_parameters, epoch_time
from rs_dataset import *
from evaluate import evaluate
from config import model_config, hyperparameters, training_config, data_config
from train import train
import numpy as np
import random
import time
import torch.optim as optim
# import GPUtil as GPU


# Optional: check available compute
# GPUs = GPU.getGPUs()
# gpu = GPUs[0] # I only have one GPU
# printm(gpu)

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset configuration
# dsets = ["lastfm-dataset-1K", "yelp", "4square", "ml-25m"]
dset = "ml-25m"
data_root = data_config["data_root"]
dset_pth = os.path.join(data_root, dset)

# Paths for training, validation, and testing
file_names = ['train.src', 'train.trg', 'val.src', 'val.trg', 'test.src', 'test.trg']
train_src, train_trg, val_src, val_trg, test_src, test_trg = [
    os.path.join(dset_pth, file_name) for file_name in file_names
]

# Path for saving the model
model_save_pth = os.path.join(training_config['out'], f"holic-{dset}.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

# model_config
enc_emb_dim = model_config["enc_emb_dim"]
dec_emb_dim = model_config["dec_emb_dim"]
enc_hid_dim = model_config["enc_hid_dim"]
dec_hid_dim = model_config["dec_hid_dim"]
enc_dropout = model_config["enc_dropout"]
dec_dropout = model_config["dec_dropout"]
src_pad_id = model_config["src_pad_id"]

# hyperparameters
k = hyperparameters["k"]
gamma = hyperparameters["gamma"]

# training_config
learning_rate = training_config["learning_rate"]
batch_size = training_config["batch_size"]
epochs = training_config["epochs"]
optimizer = training_config["optimizer"]
clip = training_config["clip"]
criterion = nn.CrossEntropyLoss()

# Instantiate the tokenizer
tokenizer = Tokenizer(dset_root=dset_pth, device=device)
input_dim = len(tokenizer.index2item)
output_dim = len(tokenizer.index2item)

# Create dataset objects for training, validation, and testing
train_dataset = InteractionDataset(src_path=train_src, trg_path=train_trg, tokenizer=tokenizer)
val_dataset = InteractionDataset(src_path=val_src, trg_path=val_trg, tokenizer=tokenizer)
test_dataset = InteractionDataset(src_path=test_src, trg_path=test_trg, tokenizer=tokenizer)

# Create DataLoaders with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

attn = Attention(enc_hid_dim, dec_hid_dim)
clusters = ClusteringLayer(k, enc_hid_dim).to(device)
enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)
model = HolicModel(enc, dec, src_pad_id, device).to(device)
model.apply(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')

model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    clusters=clusters,
    criterion=criterion,
    max_grad_norm=clip,
    epochs=100,
    patience=5,
    model_save_pth=model_save_pth,
    gamma=1.0
)