import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch

# Define default tokens
PAD_token = 0  # Padding token
SOS_token = 1  # Start-of-sequence token
EOS_token = 2  # End-of-sequence token
SOP_token = 3  # Start-of-prediction token

class Tokenizer:
    def __init__(self, dset_root, device):
        """
        Initialize the Tokenizer.

        Args:
            directory_path (str): Path to the directory containing `.src` and `.trg` files.
            device (torch.device): Device to store the tokenized data (e.g., "cpu" or "cuda").
        """
        self.directory_path = dset_root
        self.item2index = {}
        self.index2item = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            SOP_token: "SOP"
        }
        self.num_items = len(self.index2item)
        self.device = device

        # Automatically build the vocabulary from the directory
        self.build_vocab()

    def add_sequence(self, seq):
        """Add a sequence of items to the vocabulary."""
        for item in seq.split():
            self.add_item(item.strip())

    def add_item(self, item):
        """Add an item to the vocabulary."""
        if item not in self.item2index:
            self.item2index[item] = self.num_items
            self.index2item[self.num_items] = item
            self.num_items += 1

    def build_vocab(self):
        """
        Build vocabulary from all `.src` and `.trg` files in the specified directory.
        """
        src_trg_files = [os.path.join(self.directory_path, f)
                         for f in os.listdir(self.directory_path)
                         if f.endswith('.src') or f.endswith('.trg')]

        for file_path in src_trg_files:
            with open(file_path, 'r') as f:
                for line in f:
                    self.add_sequence(line)

        print(f"Vocabulary built with {self.num_items} items.")

    def tokenize_sequence(self, seq):
        """
        Tokenize a sequence into a tensor of numerical IDs.

        Args:
            seq (str): Space-delimited sequence of numerical IDs.

        Returns:
            torch.Tensor: Tokenized sequence including SOS and EOS tokens.
        """
        tokens = [SOS_token] + [self.item2index[item.strip()] for item in seq.split()] + [EOS_token]
        return torch.tensor(tokens, dtype=torch.long).to(self.device)

    def tokenize_target(self, target):
        """
        Tokenize a target into a tensor, including the SOP token.

        Args:
            target (str): A single numerical ID as a string.

        Returns:
            torch.Tensor: Tensor containing SOP and the tokenized target.
        """
        return torch.tensor([SOP_token, self.item2index[target.strip()]], dtype=torch.long).to(self.device)

class InteractionDataset(Dataset):
    def __init__(self, src_path, trg_path, tokenizer):
        """
        Args:
            src_path (str): Path to the file containing the input sequences (X).
            trg_path (str): Path to the file containing the target items (Y).
            tokenizer (Tokenizer): Tokenizer object for dynamic tokenization.
        """
        self.src_data = self._load_file(src_path)
        self.trg_data = self._load_file(trg_path)
        self.tokenizer = tokenizer

    def _load_file(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        """
        Dynamically tokenize a single example at index `idx`.

        Args:
            idx (int): Index of the data item.

        Returns:
            src_tensor (torch.Tensor): Tokenized source sequence.
            trg_tensor (torch.Tensor): Tokenized target item.
        """
        src_seq = self.src_data[idx]
        trg_item = self.trg_data[idx]
        src_tensor = self.tokenizer.tokenize_sequence(src_seq)
        trg_tensor = self.tokenizer.tokenize_target(trg_item)
        return src_tensor, trg_tensor


def collate_fn(batch):
    """
    Custom collate function to sort and pad sequences within a batch.
    """
    src_tensors, trg_tensors = zip(*batch)
    src_lengths = torch.tensor([len(src) for src in src_tensors])

    # Sort by source sequence lengths (descending order)
    sorted_indices = torch.argsort(src_lengths, descending=True)
    src_tensors = [src_tensors[i] for i in sorted_indices]
    trg_tensors = [trg_tensors[i] for i in sorted_indices]
    src_lengths = src_lengths[sorted_indices]

    # Pad source sequences to the maximum length in the batch
    padded_src = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=False, padding_value=0)

    # Combine target tensors into a single tensor
    trg_tensors = torch.stack(trg_tensors, dim=0)

    return padded_src, src_lengths, trg_tensors
