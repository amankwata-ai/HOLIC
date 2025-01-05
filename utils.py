import torch
import torch.nn as nn
# import psutil
# import humanize
import os

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# base init
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

# def init_weights(m):
#     for name, param in m.named_parameters():
#         nn.init.uniform_(param.data, -0.08, 0.08)


# def init_weights(m):
#     for name, param in m.named_parameters():
#         if 'weight' in name:
#             if 'rnn' in name:
#                 nn.init.orthogonal_(param.data)  # Better for RNN weights
#             else:
#                 nn.init.xavier_uniform_(param.data)
#         else:
#             nn.init.constant_(param.data, 0)


# def init_weights(m):
#     for name, param in m.named_parameters():
#         if 'weight' in name:
#             if 'rnn' in name or 'lstm' in name or 'gru' in name:
#                 # Orthogonal initialization works well with RNNs
#                 nn.init.orthogonal_(param.data, gain=5 / 3)  # 5/3 is gain for tanh
#
#                 # Clip weights to prevent extreme values
#                 with torch.no_grad():
#                     param.data.clamp_(-1, 1)
#
#             elif 'conv' in name:
#                 if len(param.shape) >= 2:
#                     # Kaiming initialization with tanh
#                     nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='tanh')
#                 else:
#                     # Fallback to xavier with tanh gain for 1D
#                     gain = nn.init.calculate_gain('tanh')  # Default gain for tanh is 5/3
#                     nn.init.xavier_uniform_(param.data, gain=gain)
#
#             else:
#                 # Handle different tensor dimensions for fully connected layers
#                 if len(param.shape) >= 2:
#                     gain = nn.init.calculate_gain('tanh')  # Default gain for tanh is 5/3
#                     nn.init.xavier_uniform_(param.data, gain=gain)
#                 else:
#                     # For 1D tensors, use simpler initialization scaled for tanh
#                     std = math.sqrt(2.0 / (param.shape[0] * (5 / 3)))  # Adjusted for tanh gain
#                     nn.init.uniform_(param.data, -std, std)
#
#         elif 'bias' in name:
#             # Initialize biases to zero for tanh
#             nn.init.constant_(param.data, 0.0)
#
#     # For RNN/LSTM/GRU, also ensure the hidden-to-hidden matrices are initialized properly
#     if hasattr(m, 'hidden_size'):
#         for name, param in m.named_parameters():
#             if 'weight_hh' in name:
#                 # Initialize hidden-to-hidden weights with tanh-adjusted orthogonal matrix
#                 nn.init.orthogonal_(param.data, gain=5 / 3)
#                 with torch.no_grad():
#                     param.data.clamp_(-1, 1)

def printm(gpu):
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                                gpu.memoryUsed,
                                                                                                gpu.memoryUtil * 100,
                                                                                                gpu.memoryTotal))