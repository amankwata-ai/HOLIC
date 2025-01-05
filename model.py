import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_emb = nn.LayerNorm(enc_hid_dim * 2)
        self.layer_norm_rnn_output = nn.LayerNorm(enc_hid_dim * 2)
        self.layer_norm_rnn_hidden = nn.LayerNorm(enc_hid_dim * 2)

    def forward(self, src, src_len):
        """
        Forward pass of the encoder.

        Args:
            src (torch.Tensor): Input sequence, shape (src_len, batch_size).
            src_len (torch.Tensor): Sequence lengths, shape (batch_size).

        Returns:
            outputs (torch.Tensor): Encoder outputs, shape (src_len, batch_size, enc_hid_dim * 2).
            hidden (torch.Tensor): Initial decoder hidden state, shape (batch_size, dec_hid_dim).
        """
        # Embedding and dropout
        # print(f"src shape: {src.shape}, min: {src.min()}, max: {src.max()}")
        # print(f"Embedding num_embeddings: {self.embedding.num_embeddings}")
        embedded = self.dropout(self.embedding(src))  # Shape: (src_len, batch_size, emb_dim)
        embedded = self.layer_norm_emb(embedded)  # Add normalization after embedding

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)

        # Pass through GRU
        packed_outputs, hidden = self.rnn(packed_embedded)

        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)  # Shape: (src_len, batch_size, enc_hid_dim * 2)
        outputs = self.layer_norm_rnn_output(outputs)  # Add normalization after RNN

        # Combine forward and backward hidden states from the last layer
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Shape: (batch_size, enc_hid_dim * 2)
        hidden = self.layer_norm_rnn_hidden(hidden)  # Add normalization before final FC
        hidden = torch.tanh(self.fc(hidden))  # Shape: (batch_size, dec_hid_dim)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.randn(dec_hid_dim))  # Initialized with standard normal distribution

    def forward(self, hidden, encoder_outputs, mask):
        """
        Compute the attention scores over the encoder outputs.

        Args:
            hidden (torch.Tensor): The decoder hidden state, shape (batch_size, dec_hid_dim).
            encoder_outputs (torch.Tensor): Encoder outputs, shape (src_len, batch_size, enc_hid_dim * 2).
            mask (torch.Tensor): Binary mask for padding, shape (batch_size, src_len).

        Returns:
            torch.Tensor: Attention scores, shape (batch_size, src_len).
        """
        # Transpose encoder_outputs to (batch_size, src_len, enc_hid_dim * 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Expand decoder hidden state to match encoder_outputs (batch_size, src_len, dec_hid_dim)
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).expand(-1, src_len, -1)

        # Concatenate hidden state and encoder outputs, then apply the attention layer
        combined = torch.cat((hidden, encoder_outputs), dim=2)  # Shape: (batch_size, src_len, enc_hid_dim * 2 + dec_hid_dim)
        energy = torch.tanh(self.attn(combined))  # Shape: (batch_size, src_len, dec_hid_dim)

        # Compute scores using self.v, avoid redundant repeat operation
        attention = torch.einsum("bld,d->bl", energy, self.v)  # Shape: (batch_size, src_len)

        # Apply mask to ignore padded elements
        attention = attention.masked_fill(mask == 0, -1e10)

        # Normalize scores using softmax
        return F.softmax(attention, dim=1)


class ClusteringLayer(nn.Module):
    """
    A PyTorch layer for clustering, converting input features to soft labels using Student's t-distribution.

    Args:
        n_clusters (int): Number of clusters.
        enc_hid_dim (int): Dimensionality of the encoder hidden representation.
        weights (torch.Tensor, optional): Initial cluster centers with shape `(n_clusters, enc_hid_dim)`.
        alpha (float, optional): Parameter for Student's t-distribution. Default is 1.0.
    """
    def __init__(self, n_clusters, enc_hid_dim, weights=None, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Initialize cluster weights
        self.cluster_weights = nn.Parameter(
            torch.Tensor(n_clusters, enc_hid_dim)
        )
        if weights is not None:
            self.cluster_weights.data = torch.tensor(weights, dtype=torch.float32)
        else:
            nn.init.xavier_normal_(self.cluster_weights)

    def forward(self, enc_hidden):
        """
        Forward pass: Calculates the Student's t-distribution (soft labels) for each sample.

        Args:
            enc_hidden (torch.Tensor): Input features, shape `(n_samples, n_features)`.

        Returns:
            torch.Tensor: Soft labels, shape `(n_samples, n_clusters)`.
        """
        # Compute pairwise squared distances between inputs and cluster centers
        dist = torch.sum((enc_hidden.unsqueeze(1) - self.cluster_weights) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q /= torch.sum(q, dim=1, keepdim=True)  # Normalize across clusters

        return q

    @staticmethod
    def target_distribution(q):
        """
        Computes the target distribution used to refine clusters.

        Args:
            q (torch.Tensor): Soft labels, shape `(n_samples, n_clusters)`.

        Returns:
            torch.Tensor: Refined target distribution, shape `(n_samples, n_clusters)`.
        """
        weight = q ** 2 / q.sum(dim=0)
        return (weight.t() / weight.sum(dim=1)).t()

    def update_centroids(self, new_centroids):
        """
        Updates the cluster centers with new values.

        Args:
            new_centroids (torch.Tensor): New cluster centers, shape `(n_clusters, n_features)`.
        """
        self.cluster_weights.data = new_centroids

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=False)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        """
        Forward pass for the decoder.

        Args:
            input (torch.Tensor): <sos> token, shape (batch_size).
            hidden (torch.Tensor): Encoder final hidden state, shape (batch_size, dec_hid_dim).
            encoder_outputs (torch.Tensor): Encoder outputs, shape (src_len, batch_size, enc_hid_dim * 2).
            mask (torch.Tensor): Source sequence mask, shape (batch_size, src_len).

        Returns:
            prediction (torch.Tensor): Predicted token scores, shape (batch_size, output_dim).
        """
        # Embed the input token
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))  # Shape: (1, batch_size, emb_dim)

        # Compute attention weights
        attention_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)  # Shape: (batch_size, 1, src_len)

        # Compute weighted sum of encoder outputs
        weighted = torch.bmm(attention_weights, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        # weighted shape: (1, batch_size, enc_hid_dim * 2)

        # Prepare input for the RNN
        rnn_input = torch.cat((embedded, weighted), dim=2)  # Shape: (1, batch_size, (enc_hid_dim * 2) + emb_dim)

        # Pass through the GRU
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output shape: (1, batch_size, dec_hid_dim), hidden shape: (1, batch_size, dec_hid_dim)

        # Final prediction
        output, embedded, weighted = output.squeeze(0), embedded.squeeze(0), weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # Shape: (batch_size, output_dim)

        return prediction, hidden.squeeze(0), attention_weights.squeeze(1)



class HolicModel(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        """
        Create a mask for the source sequence.

        Args:
            src (torch.Tensor): Source sequence, shape (src_len, batch_size).

        Returns:
            torch.Tensor: Mask, shape (batch_size, src_len).
        """
        return (src != self.src_pad_idx).permute(1, 0)

    def forward(self, src, src_len, trg):
        """
        Forward pass for HolicModel.

        Args:
            src (torch.Tensor): Source sequence, shape (src_len, batch_size).
            src_len (torch.Tensor): Lengths of source sequences, shape (batch_size).
            trg (torch.Tensor): Target sequence, shape (2, batch_size). (e.g., [<sop>, target_item])

        Returns:
            prediction (torch.Tensor): Predicted token scores, shape (batch_size, trg_vocab_size).
            enc_hidden (torch.Tensor): Final encoder hidden state, shape (batch_size, dec_hid_dim).
        """
        # Encoder forward pass
        encoder_outputs, hidden = self.encoder(src, src_len)

        # Extract <sop> token as the input
        # input = trg[0, :]
        input = trg[:, 0]

        # Create mask for the source sequence
        mask = self.create_mask(src)

        # Decoder forward pass for a single step
        prediction, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

        return prediction, hidden
