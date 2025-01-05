import os
import sys
import zipfile
from config import data_config
import pandas as pd
import random
import json
import tarfile
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter


def extract_file(source_dir: str, file_name: str, destination_dir: str):
    """
    Extracts a compressed file (zip or tar) from the source directory to the destination directory.

    Parameters:
        source_dir (str): The directory containing the compressed file.
        file_name (str): The name of the compressed file (must include the extension).
        destination_dir (str): The base directory where the extracted files will be placed.

    Prints:
        Success message if extraction is successful.
        Error message if the operation fails.
    """
    # Construct the full path of the compressed file
    file_path = os.path.join(source_dir, file_name)

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check the file extension and extract accordingly
        if zipfile.is_zipfile(file_path):
            # Handle .zip files
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_dir)
            print(f"Successfully extracted '{file_name}' as a ZIP to '{destination_dir}'.")
        elif tarfile.is_tarfile(file_path):
            # Handle .tar files
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(destination_dir)
            print(f"Successfully extracted '{file_name}' as a TAR to '{destination_dir}'.")
        else:
            raise ValueError(f"Unsupported file type or invalid archive: {file_name}")
    except Exception as e:
        print(f"Failed to extract '{file_name}': {e}")


def read_file_as_dataframe(file_path, header=None, delimiter=None, encoding="utf-8"):
    """
    Reads a file as a pandas DataFrame, handling text or other delimited file types.

    Parameters:
        file_path (str): Path to the input file.
        delimiter (str): Delimiter for line-separated values (e.g., ',' for CSV, '\t' for TSV). Defaults to None for auto-detection.
        encoding (str): File encoding. Defaults to 'utf-8'.

    Returns:
        pd.DataFrame: The DataFrame containing the parsed data.
    """
    try:
        if file_path.endswith('.csv'):
            # Read as CSV
            return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        elif file_path.endswith(('.txt', '.tsv')):
            # Read as a text file with a specified delimiter
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            # Assume the first line contains headers, split on delimiter
            # header = lines[0].strip().split(delimiter) if delimiter else lines[0].strip().split()
            data = [line.strip().split(delimiter) if delimiter else line.strip().split() for line in lines]

            # check data consistency
            lengths = [len(value) for value in data]
            print(f"{min(lengths)}, {max(lengths)}")

            return pd.DataFrame(data, columns=header)
        else:
            raise ValueError(f"Unsupported file format for '{file_path}'.")
    except Exception as e:
        print(f"Failed to read file '{file_path}': {e}")
        return pd.DataFrame()


def generate_sequences(
        extract_dir: str,
        target_file: str,
        column_mapping: Dict[str, str],
        delimiter: str,
        max_length: int,
        min_frequency: int = 0,
        header: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Process an interaction table file to map user IDs to sorted lists of item IDs.
    Long sequences are split into multiple shorter sequences.

    Args:
        extract_dir: Directory containing the target file
        target_file: Name of the CSV file to process
        column_mapping: Dictionary mapping generic column names to actual dataset column names
            Required keys: "userid", "itemid", "timestamp"
        delimiter: CSV delimiter character
        max_length: Maximum length of item list for each sequence
        min_frequency: Minimum number of times an item must appear to be included.
                      If 0, all items are included without frequency checking.
        header: Optional list of column names. If None, uses first row as header

    Returns:
        Dictionary mapping user IDs to timestamp-sorted item ID lists.
        User IDs for split sequences are suffixed with _1, _2, etc.

    Raises:
        ValueError: If column mapping is invalid or required columns are missing
    """
    try:
        # Validate required column mappings
        required_keys = {"userid", "itemid", "timestamp"}
        if not required_keys.issubset(column_mapping.keys()):
            missing = required_keys - set(column_mapping.keys())
            raise ValueError(f"Column mapping missing required keys: {missing}")

        # Load the CSV file
        file_path = os.path.join(extract_dir, target_file)
        if header is None:
            df = pd.read_csv(file_path)[column_mapping.values()]
        else:
            df = read_file_as_dataframe(
                file_path,
                header,
                delimiter=delimiter,
                encoding="utf-8"
            )[column_mapping.values()]

        # Validate all mapped columns exist in the dataset
        missing_cols = set(column_mapping.values()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV file: {missing_cols}")

        # Clean data by removing rows with empty item IDs
        df = df.replace('', np.nan).dropna(subset=[column_mapping['itemid']])

        # Filter items if frequency threshold is specified
        if min_frequency > 0:
            item_counts = df[column_mapping['itemid']].value_counts()
            frequent_items = item_counts[item_counts >= min_frequency].index
            df = df[df[column_mapping['itemid']].isin(frequent_items)]

        # Sort by timestamp and group by user
        grouped = (df.sort_values(by=column_mapping['timestamp'])
                   .groupby(column_mapping['userid'])[column_mapping['itemid']]
                   .agg(list))

        # Split sequences and create new dictionary
        user_item_sequences = {}
        for user_id, items in grouped.items():
            # If sequence is shorter than max_length, keep as is
            if len(items) <= max_length:
                user_item_sequences[user_id] = items
                continue

            # Split longer sequences
            num_sequences = (len(items) + max_length - 1) // max_length
            for i in range(num_sequences):
                start_idx = i * max_length
                end_idx = min((i + 1) * max_length, len(items))
                sequence = items[start_idx:end_idx]

                # Generate new user ID with suffix for split sequences
                new_user_id = f"{user_id}_{i + 1}" if i > 0 else user_id
                user_item_sequences[new_user_id] = sequence

        return user_item_sequences

    except Exception as e:
        print(f"Error processing file '{target_file}': {str(e)}")
        return {}


def create_datasets(
        dset: str,
        user_sequences: dict,
        # destination_dir: str,
        train_test_split: float
):
    """
    Creates train, validation, and test datasets from user sequences.

    Parameters:
        user_sequences (dict): A dictionary mapping user IDs to item ID sequences.
        destination_dir (str): Directory where the datasets will be saved.
        train_test_split (float): Proportion of the dataset to use for validation and testing.

    Returns:
        tuple: Six lists (train_src, train_trg, val_src, val_trg, test_src, test_trg).
    """
    if len(user_sequences) == 0:
        raise ValueError("user_sequences cannot be empty!")

    if not (0 < train_test_split < 1):
        raise ValueError("train_test_split must be a float between 0 and 1.")

    destination_dir = os.path.join(data_config["processed"], dset)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Prepare the dataset
    src = []
    trg = []
    for user, items in user_sequences.items():
        if len(items) < 2:  # Skip if not enough items to create src and trg 2
            continue
        src.append(" ".join(map(str, items[:-1])))  # Concatenate all but the last item
        trg.append(str(items[-1]))  # The last item as target

    # Combine src and trg into a single dataset
    dataset = list(zip(src, trg))
    random.shuffle(dataset)  # Shuffle the dataset

    # Split the dataset
    val_test_size = int(len(dataset) * train_test_split)
    val_test_data = dataset[:val_test_size]
    train_data = dataset[val_test_size:]

    # Split validation and test sets from the val_test_data
    val_size = len(val_test_data) // 2
    val_data = val_test_data[:val_size]
    test_data = val_test_data[val_size:]

    # Unpack src and trg from the splits
    train_src, train_trg = zip(*train_data) if train_data else ([], [])
    val_src, val_trg = zip(*val_data) if val_data else ([], [])
    test_src, test_trg = zip(*test_data) if test_data else ([], [])

    # Convert to lists
    train_src, train_trg = list(train_src), list(train_trg)
    val_src, val_trg = list(val_src), list(val_trg)
    test_src, test_trg = list(test_src), list(test_trg)

    # Save to files in the destination directory
    def save_to_file(data, file_name):
        with open(os.path.join(destination_dir, file_name), "w") as f:
            f.writelines("\n".join(data) + "\n")

    # dset_dir = os.path.join(data_config["processed"], dset)
    save_to_file(train_src, os.path.join(destination_dir, "train.src"))
    save_to_file(train_trg, os.path.join(destination_dir, "train.trg"))
    save_to_file(val_src, os.path.join(destination_dir, "val.src"))
    save_to_file(val_trg, os.path.join(destination_dir, "val.trg"))
    save_to_file(test_src, os.path.join(destination_dir, "test.src"))
    save_to_file(test_trg, os.path.join(destination_dir, "test.trg"))

    print(f"Successfully created '{dset}' dataset.")


def generate_sequences_from_json_lines(
        extract_dir: str,
        target_file: str,
        column_mapping: Dict[str, str],
        max_length: int,
        min_frequency: int = 0
) -> Dict[str, List[str]]:
    """
    Process a JSON file with one JSON object per line to create a map of user IDs to sorted lists of item IDs.
    Long sequences are split into multiple shorter sequences.

    Args:
        extract_dir: Directory containing the target file
        target_file: Name of the JSON file to process
        column_mapping: Dictionary mapping generic column names to JSON object keys
            Required keys: "userid", "itemid", "timestamp"
        max_length: Maximum length of item list for each sequence
        min_frequency: Minimum number of times an item must appear to be included

    Returns:
        Dictionary mapping user IDs to timestamp-sorted item ID lists.
        User IDs for split sequences are suffixed with _1, _2, etc.

    Raises:
        ValueError: If column mapping is invalid or required keys are missing
    """
    # Validate the column_mapping keys
    required_keys = ["userid", "itemid", "timestamp"]
    for key in required_keys:
        if key not in column_mapping:
            raise ValueError(f"Column mapping must include '{key}'")

    file_path = os.path.join(extract_dir, target_file)

    def split_user_sequences(sequences: List[Tuple[str, str]], user_id: str) -> Dict[str, List[str]]:
        """Helper function to split sequences and create user mappings."""
        result = {}
        sorted_items = [item for _, item in sorted(sequences)]

        if len(sorted_items) <= max_length:
            result[user_id] = sorted_items
        else:
            num_sequences = (len(sorted_items) + max_length - 1) // max_length
            for i in range(num_sequences):
                start_idx = i * max_length
                end_idx = min((i + 1) * max_length, len(sorted_items))
                new_user_id = f"{user_id}_{i + 1}" if i > 0 else user_id
                result[new_user_id] = sorted_items[start_idx:end_idx]

        return result

    try:
        # Fast path when no frequency filtering is needed
        if min_frequency <= 0:
            temp_sequences = {}
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    json_obj = json.loads(line.strip())
                    user_id = json_obj[column_mapping["userid"]]
                    item_id = json_obj[column_mapping["itemid"]]
                    timestamp = json_obj[column_mapping["timestamp"]]

                    if user_id not in temp_sequences:
                        temp_sequences[user_id] = []
                    temp_sequences[user_id].append((timestamp, item_id))

            # Split and process sequences
            result = {}
            for user_id, sequences in temp_sequences.items():
                result.update(split_user_sequences(sequences, user_id))
            return result

        # Normal path with frequency filtering
        item_counter = Counter()

        # First pass: Count item frequencies
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line.strip())
                item_id = json_obj[column_mapping["itemid"]]
                item_counter[item_id] += 1

        # Get items that meet minimum frequency
        frequent_items = {item for item, count in item_counter.items()
                          if count >= min_frequency}

        # Second pass: Create sequences with filtered items
        temp_sequences = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line.strip())

                user_id = json_obj[column_mapping["userid"]]
                item_id = json_obj[column_mapping["itemid"]]
                timestamp = json_obj[column_mapping["timestamp"]]

                if item_id in frequent_items:
                    if user_id not in temp_sequences:
                        temp_sequences[user_id] = []
                    temp_sequences[user_id].append((timestamp, item_id))

        # Split and process sequences
        result = {}
        for user_id, sequences in temp_sequences.items():
            if sequences:  # Only process users with items
                result.update(split_user_sequences(sequences, user_id))

        return result

    except Exception as e:
        print(f"Failed to process the JSON file '{file_path}': {e}")
        return {}





def process_data(dset: str, data_config: Dict[str, Any]) -> None:
    """
    Process dataset files and generate sequences based on dataset type.

    Args:
        dset: Dataset name ('yelp', '4square', or 'ml-25m')
        data_config: Configuration dictionary containing dataset parameters
    """
    # Check if dataset file exists
    dataset_path = os.path.join(data_config["raw"], data_config[dset]["filename"])
    if not os.path.exists(dataset_path):
        print(f"""
                Dataset has to be downloaded manually:
                1. Download the zip folder from {data_config[dset]["url"]}
                2. Save it to {data_config["raw"]}
                3. Rerun preprocess_data
                """)
        sys.exit(1)

    # Extract dataset files
    extract_file(
        data_config["raw"],
        data_config[dset]["filename"],
        data_config[dset]["raw_dest"]
    )

    # Generate sequences based on dataset type
    dataset_folder = os.path.join(data_config["raw"], dset)
    target = data_config[dset]["targets"][0]
    columns = data_config[dset]["columns"]
    max_length = data_config["max_length"]
    min_frequency = data_config["min_frequency"]

    if dset == "yelp":
        sequences = generate_sequences_from_json_lines(
            dataset_folder,
            target,
            columns,
            max_length,
            min_frequency
        )
    else:  # 4square or ml-25m
        delimiter = data_config[dset]["delimeter"]
        header = data_config[dset].get("header", None)  # Default to None if not specified

        sequences = generate_sequences(
            dataset_folder,
            target,
            columns,
            delimiter,
            max_length,
            min_frequency,
            header
        )

    # Create train/test splits
    create_datasets(dset, sequences, train_test_split=0.4)


process_data(dset="ml-25m", data_config=data_config)

# dsets = ["lastfm-dataset-1K", "yelp", "4square", "ml-25m"]
# for dset in dsets:
#     process_data(dset=dset, data_config=data_config)
