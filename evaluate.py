import torch


def get_recall(indices, targets, k=20):
    """
    Calculates the recall score for the given predictions and targets.

    Args:
        indices (Tensor): Shape (B, k). Top-k indices predicted by the model.
        targets (Tensor): Shape (B). Actual target indices.
        k (int): The number of top recommendations to consider. Default is 20.

    Returns:
        float: The recall score, i.e., the proportion of targets found in the top-k predictions.
    """
    # Expand targets to match the dimensions of indices
    targets = targets.unsqueeze(1)  # Shape: (B, 1)

    # Check if targets are within the top-k indices
    hits = (targets == indices).any(dim=1)  # Shape: (B,)

    # Calculate recall as the mean of hits
    recall = hits.float().mean().item()

    return recall


def get_mrr(indices, targets, k=20):
    """
    Calculates the Mean Reciprocal Rank (MRR) score for the given predictions and targets.

    Args:
        indices (Tensor): Shape (B, k). Top-k indices predicted by the model.
        targets (Tensor): Shape (B). Actual target indices.
        k (int): The number of top recommendations to consider. Default is 20.

    Returns:
        float: The MRR score.
    """
    # Expand targets to match the shape of indices
    targets = targets.unsqueeze(1)  # Shape: (B, 1)

    # Find the positions of hits in the top-k indices
    matches = (indices == targets).nonzero(as_tuple=False)

    if matches.numel() == 0:  # Handle the case where no matches are found
        return 0.0

    # Extract the rank of the first match for each target
    ranks = matches[:, -1].float() + 1  # Convert to 1-based index

    # Calculate reciprocal ranks
    reciprocal_ranks = torch.reciprocal(ranks)

    # Calculate the mean reciprocal rank
    mrr = reciprocal_ranks.sum().item() / targets.size(0)

    return mrr

def evaluate_metrics(indices, targets, k=20, ignore_index=3):
    """
    Evaluates the model using Recall@K and MRR@K metrics.

    Args:
        indices (Tensor): Shape (B, C). Logits or predicted scores for the next items.
        targets (Tensor): Shape (B). Actual target indices.
        k (int): The number of top recommendations to consider for evaluation. Default is 20.
        ignore_index (int): Target value to ignore for valid calculations. Default is 3.

    Returns:
        tuple: Recall@K, MRR@K, Valid MRR@K, Valid Recall@K scores.
    """
    # Extract top-k indices from the logits
    top_k_indices = torch.topk(indices, k, dim=-1).indices

    # Compute metrics
    recall = get_recall(top_k_indices, targets)
    mrr = get_mrr(top_k_indices, targets)

    return recall, mrr


def evaluate(model, iterator, criterion):
    """
    Evaluate the model's performance on the provided dataset iterator.

    Args:
        model (nn.Module): The sequence-to-sequence model.
        iterator (DataLoader): DataLoader providing batches of input-output pairs.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        tuple: Average loss, MRR, recall, valid MRR, and valid recall for the dataset.
    """
    model.eval()

    # Initialize metrics
    epoch_metrics = {"loss": 0, "mrr": 0, "recall": 0}
    num_batches = len(iterator)

    with torch.no_grad():
        for src, src_len, trg in iterator:

            # get output and target
            output, _ = model(src, src_len, trg)
            trg = trg[:, 1]

            # Calculate loss
            loss = criterion(output, trg)

            # Compute metrics
            recall, mrr, = evaluate_metrics(output, trg)

            # Update metrics
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["mrr"] += mrr
            epoch_metrics["recall"] += recall

    # Compute average metrics
    avg_metrics = {key: value / num_batches for key, value in epoch_metrics.items()}

    return (
        avg_metrics["loss"],
        avg_metrics["mrr"],
        avg_metrics["recall"],
    )