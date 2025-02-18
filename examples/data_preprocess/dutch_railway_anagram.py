"""
Preprocess dataset for anagram task - given a scrambled Dutch railway station name, unscramble it to find the original station.
Ensures test set contains some unique station names not seen in training.
"""

import re
import os
from datasets import Dataset
from random import shuffle, seed, choice, sample
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
from collections import Counter

# Category description matching the format of the general script
CATEGORY_DESCRIPTIONS = {
    'station_list.txt': 'a Dutch railway station name'
}

def is_scrambleable(word: str) -> bool:
    """Check if a word can be scrambled (has at least 2 different characters)."""
    if not word:
        return False
    char_counts = Counter(word.lower().replace(" ", ""))
    # Word must have at least 2 different characters to be scrambleable
    return len(char_counts) >= 2

def scramble_word(word: str, max_attempts: int = 10) -> Optional[str]:
    """Scramble a word while ensuring it's different from the original.
    Spaces are removed from input, but output has spaces between characters.
    Returns None if scrambling is impossible or fails after max attempts."""
    try:
        word = word.lower().strip()
        word_no_spaces = word.replace(" ", "")
        
        # Check if word can be scrambled
        if not is_scrambleable(word_no_spaces):
            return None
            
        word_list = list(word_no_spaces)
        for _ in range(max_attempts):
            shuffle(word_list)
            scrambled = ' '.join(word_list)
            if scrambled.replace(" ", "") != word_no_spaces:
                return scrambled
        return None
    except Exception as e:
        print(f"Error scrambling word '{word}': {str(e)}")
        return None

def split_words_for_test(
    words: List[str],
    test_ratio: float = 0.2,
    seed_value: int = 42
) -> Dict[str, List[str]]:
    """Split words into completely separate training and test sets."""
    seed(seed_value)
    num_test_words = int(len(words) * test_ratio)
    test_words = sample(words, num_test_words)
    train_words = [word for word in words if word not in test_words]
    return {'train': train_words, 'test': test_words}

def gen_dataset(
    word_splits: Dict[str, List[str]],
    num_samples: int,
    category_name: str = "",
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset samples, skipping words that can't be scrambled."""
    seed(seed_value)
    samples = []
    skipped_words = set()
    
    with tqdm(total=num_samples, desc=f"Generating samples for {category_name}") as pbar:
        while len(samples) < num_samples:
            target_word = choice(word_splits['train'])
            scrambled = scramble_word(target_word)
            
            if scrambled is None:
                if target_word not in skipped_words:
                    skipped_words.add(target_word)
                    print(f"Warning: Skipped unscrambleable word '{target_word}' in {category_name}")
                continue
                
            samples.append((scrambled, target_word))
            pbar.update(1)
    
    if skipped_words:
        print(f"\nSkipped {len(skipped_words)} unscrambleable words in {category_name}")
        
    return samples

def make_prefix(dp, template_type: str = 'llama-instruct') -> str:
    """Generate the prompt prefix based on the category and template type."""
    scrambled_word = dp['scrambled_word']
    category_desc = CATEGORY_DESCRIPTIONS['station_list.txt']
    
    return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant. You first think about the reasoning process out loud and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

The following scrambled characters make up the letters of {category_desc} (like an anagram): "{scrambled_word}". Spaces have been added between the scrambled letters for improved legibility, but any spaces in the original name have been removed. Show your reasoning in <think> </think> tags. Once you have thought about it, put your answer between <answer> and </answer> tags.<|eot_id|>'''

if __name__ == '__main__':
    print("Starting station anagram generation script...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/railway_anagram')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--template_type', type=str, default='llama-instruct')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Load Dutch railway stations from file
    with open('data/station_list.txt', 'r') as f:
        words = [line.strip() for line in f.readlines()
                if len(line.strip().replace(" ", "")) >= 3 
                and is_scrambleable(line.strip())]
        print(f"Found {len(words)} valid station names")

    # Split stations into training and test sets
    word_splits = split_words_for_test(
        words,
        test_ratio=args.test_ratio,
        seed_value=args.seed
    )

    # Generate training samples
    train_samples = gen_dataset(
        word_splits,
        num_samples=args.train_size,
        category_name='station_list.txt',
        seed_value=args.seed
    )
    
    # Generate test samples
    test_samples = gen_dataset(
        {'train': word_splits['test']},  # Use test words for test set
        num_samples=args.test_size,
        category_name='station_list.txt (test)',
        seed_value=args.seed
    )

    # Convert to datasets
    train_dict = {
        'scrambled_word': [sample[0] for sample in train_samples],
        'target_word': [sample[1] for sample in train_samples],
        'category': ['station_list.txt'] * len(train_samples)
    }
    
    test_dict = {
        'scrambled_word': [sample[0] for sample in test_samples],
        'target_word': [sample[1] for sample in test_samples],
        'category': ['station_list.txt'] * len(test_samples)
    }
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "scrambled_word": example['scrambled_word'],
                "target_word": example['target_word'],
                "category": example['category'],
                "all_words": words
            }
            data = {
                "data_source": "anagram_stations",
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

    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Print statistics about the datasets
    unique_train_words = set(train_dict['target_word'])
    unique_test_words = set(test_dict['target_word'])
    test_only_words = unique_test_words - unique_train_words
    
    print("\nDataset Statistics:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Unique stations in training: {len(unique_train_words)}")
    print(f"Unique stations in test: {len(unique_test_words)}")
    print(f"Stations that appear only in test: {len(test_only_words)}")
    print("Test-only stations:", sorted(list(test_only_words)))

    # Save datasets
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)