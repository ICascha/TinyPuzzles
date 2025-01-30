"""
Preprocess dataset for anagram task - given a scrambled US state name, unscramble it to find the original state.
Ensures test set contains some unique states not seen in training.
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
    """Scramble a word while ensuring it's different from the original."""
    word = word.lower().strip()
    word_list = list(word.replace(" ", ""))
    while True:
        shuffle(word_list)
        scrambled = ''.join(word_list)
        if scrambled != word.replace(" ", ""):
            return scrambled


def split_states_for_test(
    us_states: List[str],
    num_unique_test_states: int = 10,
    seed_value: int = 42
) -> Dict[str, List[str]]:
    """Split states into training and test sets, with some states reserved for testing."""
    seed(seed_value)
    
    # Randomly select states that will only appear in test set
    unique_test_states = sample(us_states, num_unique_test_states)
    
    # Remaining states can appear in both training and test
    shared_states = [state for state in us_states if state not in unique_test_states]
    
    return {
        'unique_test': unique_test_states,
        'shared': shared_states
    }


def gen_dataset(
    state_splits: Dict[str, List[str]],
    num_train_samples: int,
    num_test_samples: int,
    test_unique_ratio: float = 0.3,  # Percentage of test samples that should use unique states
    seed_value: int = 42,
) -> Dict[str, List[Tuple]]:
    """Generate dataset for anagram task with unique test states.
    
    Args:
        state_splits: Dictionary containing 'unique_test' and 'shared' state lists
        num_train_samples: Number of training samples to generate
        num_test_samples: Number of test samples to generate
        test_unique_ratio: Ratio of test samples that should use unique test states
        seed_value: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test samples
    """
    seed(seed_value)
    train_samples = []
    test_samples = []
    
    # Generate training samples (only from shared states)
    for _ in tqdm(range(num_train_samples), desc="Generating training samples"):
        target_state = choice(state_splits['shared'])
        scrambled = scramble_word(target_state)
        train_samples.append((scrambled, target_state))
    
    # Calculate number of unique state test samples
    num_unique_test = int(num_test_samples * test_unique_ratio)
    num_shared_test = num_test_samples - num_unique_test
    
    # Generate test samples from unique states
    for _ in tqdm(range(num_unique_test), desc="Generating unique test samples"):
        target_state = choice(state_splits['unique_test'])
        scrambled = scramble_word(target_state)
        test_samples.append((scrambled, target_state))
    
    # Generate remaining test samples from shared states
    for _ in tqdm(range(num_shared_test), desc="Generating shared test samples"):
        target_state = choice(state_splits['shared'])
        scrambled = scramble_word(target_state)
        test_samples.append((scrambled, target_state))
    
    return {
        'train': train_samples,
        'test': test_samples
    }


def make_prefix(dp, template_type):
    scrambled_word = dp['scrambled_word']
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Unscramble this US state name: {scrambled_word}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>California</answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nUnscramble this US state name: {scrambled_word}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>California</answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/anagram')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--num_unique_test_states', type=int, default=10)
    parser.add_argument('--test_unique_ratio', type=float, default=0.3)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # List of US states
    US_STATES = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

    # Split states into training and test sets
    state_splits = split_states_for_test(
        US_STATES,
        num_unique_test_states=args.num_unique_test_states,
        seed_value=args.seed
    )

    # Generate datasets
    datasets = gen_dataset(
        state_splits,
        num_train_samples=args.train_size,
        num_test_samples=args.test_size,
        test_unique_ratio=args.test_unique_ratio,
        seed_value=args.seed
    )
    
    data_source = 'anagram'

    # Convert to datasets
    train_dict = {
        'scrambled_word': [sample[0] for sample in datasets['train']],
        'target_state': [sample[1] for sample in datasets['train']]
    }
    
    test_dict = {
        'scrambled_word': [sample[0] for sample in datasets['test']],
        'target_state': [sample[1] for sample in datasets['test']]
    }
    
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "scrambled_word": example['scrambled_word'],
                "target_state": example['target_state'],
                "us_states": US_STATES
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
    unique_train_states = set(train_dict['target_state'])
    unique_test_states = set(test_dict['target_state'])
    test_only_states = unique_test_states - unique_train_states
    
    print("\nDataset Statistics:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Unique states in training: {len(unique_train_states)}")
    print(f"Unique states in test: {len(unique_test_states)}")
    print(f"States that appear only in test: {len(test_only_states)}")
    print("Test-only states:", sorted(list(test_only_states)))

    # Save datasets
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)