"""
Preprocess dataset for anagram task - given a scrambled Dutch railway station name, unscramble it to find the original station.
Ensures test set contains some unique station names not seen in training.
"""

import re
import os
from datasets import Dataset
from random import shuffle, seed, choice, sample
from typing import List, Tuple, Dict
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def scramble_word(word: str) -> str:
    """Scramble a word while ensuring it's different from the original.
    Spaces are removed from both input and output."""
    word = word.lower().strip()
    word_no_spaces = word.replace(" ", "")
    word_list = list(word_no_spaces)
    while True:
        shuffle(word_list)
        scrambled = ''.join(word_list)
        if scrambled != word_no_spaces:
            return scrambled


def split_stations_for_test(
    stations: List[str],
    test_ratio: float = 0.2,  # Proportion of stations to reserve for testing
    seed_value: int = 42
) -> Dict[str, List[str]]:
    """Split stations into completely separate training and test sets.
    
    Args:
        stations: List of all Dutch railway station names
        test_ratio: Proportion of stations to use for testing (default 0.2 = 20%)
        seed_value: Random seed for reproducibility
    
    Returns:
        Dictionary containing separate train and test station lists
    """
    seed(seed_value)
    
    # Calculate number of stations for test set
    num_test_stations = int(len(stations) * test_ratio)
    
    # Randomly select stations for test set
    test_stations = sample(stations, num_test_stations)
    
    # Remaining stations go to train set
    train_stations = [station for station in stations if station not in test_stations]
    
    return {
        'train': train_stations,
        'test': test_stations
    }


def gen_dataset(
    station_splits: Dict[str, List[str]],
    num_train_samples: int,
    num_test_samples: int,
    seed_value: int = 42,
) -> Dict[str, List[Tuple]]:
    """Generate dataset for anagram task with completely separate train/test stations.
    
    Args:
        station_splits: Dictionary containing separate 'train' and 'test' station lists
        num_train_samples: Number of training samples to generate
        num_test_samples: Number of test samples to generate
        seed_value: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test samples
    """
    seed(seed_value)
    train_samples = []
    test_samples = []
    
    # Generate training samples (only from train stations)
    for _ in tqdm(range(num_train_samples), desc="Generating training samples"):
        target_station = choice(station_splits['train'])
        scrambled = scramble_word(target_station)
        train_samples.append((scrambled, target_station))
    
    # Generate test samples (only from test stations)
    for _ in tqdm(range(num_test_samples), desc="Generating test samples"):
        target_station = choice(station_splits['test'])
        scrambled = scramble_word(target_station)
        test_samples.append((scrambled, target_station))
    
    return {
        'train': train_samples,
        'test': test_samples
    }


def make_prefix(dp, template_type):
    scrambled_word = dp['scrambled_word']
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Unscramble this Dutch railway station name: {scrambled_word}. Note that any spaces from the original station name have been removed. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example if the scrambled word was "damstermcntraal", the answer would be <answer>Amsterdam Centraal</answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nUnscramble this Dutch railway station name: {scrambled_word}. Note that any spaces from the original station name have been removed. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example if the scrambled word was "damstermcntraal", the answer would be <answer>Amsterdam Centraal</answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/anagram')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Load Dutch railway stations from file
    with open('data/dutch_railway_stations.txt', 'r') as f:
        STATIONS = [line.strip() for line in f.readlines()]

    # Split stations into training and test sets
    station_splits = split_stations_for_test(
        STATIONS,
        test_ratio=args.test_ratio,
        seed_value=args.seed
    )

    # Generate datasets
    datasets = gen_dataset(
        station_splits,
        num_train_samples=args.train_size,
        num_test_samples=args.test_size,
        seed_value=args.seed
    )
    
    data_source = 'anagram'

    # Convert to datasets
    train_dict = {
        'scrambled_word': [sample[0] for sample in datasets['train']],
        'target_station': [sample[1] for sample in datasets['train']]
    }
    
    test_dict = {
        'scrambled_word': [sample[0] for sample in datasets['test']],
        'target_station': [sample[1] for sample in datasets['test']]
    }
    
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "scrambled_word": example['scrambled_word'],
                "target_station": example['target_station'],
                "stations": STATIONS
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "word_unscramble",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Print statistics about the datasets
    unique_train_stations = set(train_dict['target_station'])
    unique_test_stations = set(test_dict['target_station'])
    test_only_stations = unique_test_stations - unique_train_stations
    
    print("\nDataset Statistics:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Unique stations in training: {len(unique_train_stations)}")
    print(f"Unique stations in test: {len(unique_test_stations)}")
    print(f"Stations that appear only in test: {len(test_only_stations)}")
    print("Test-only stations:", sorted(list(test_only_stations)))

    # Save datasets
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
