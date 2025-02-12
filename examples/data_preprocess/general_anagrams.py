"""
Preprocess dataset for anagram tasks - given scrambled words from various categories,
unscramble them to find the original word. Ensures test set contains some unique words
not seen in training for each category.
"""

import re
import os
from datasets import Dataset, concatenate_datasets
from random import shuffle, seed, choice, sample
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from pathlib import Path
import argparse
from collections import Counter

# Category descriptions for prompts
CATEGORY_DESCRIPTIONS = {
    'animals.txt': 'one of the 250 most commonly recognized English animal names',
    'capitals.txt': 'a capital city name of a country',
    'countries.txt': 'a country name',
    'elements.txt': 'an English chemical element name',
    'european_clubs.txt': 'the name of a top 200 ranked European football/soccer club',
    'fortune500_2023.txt': 'a Fortune 500 company name (2023)',
    'fruits.txt': 'a name of a fruit',
    'nobel_laureates_in_literature.txt': 'the name of a Nobel Prize winner in Literature from 1901-2023',
    'nobel_laureates_in_physics.txt': 'the name of a Nobel Prize winner in Physics from 1901-2023',
    'top200_languages.txt': 'one of the 200 most spoken languages in the world',
    'top300_albums.txt': 'the name of one of the 300 best-selling music albums of all time',
    'top_american_brands.txt': 'one of the 500 most recognized American brand names by current consumers',
    'top_artists.txt': 'the name of one of the 150 most commercially successful musical artists (band names included) of all time',
    'top_books.txt': 'the title of one of the 200 best-selling books of all time',
    'travel_destinations.txt': 'one of the 250 most famous travel destinations in the world (landmarks, museums, buildings, monuments, etc.)',
    'us_cities_100k.txt': 'the name of a US city with a population over 100,000',
    'vegetables.txt': 'a vegetable name'
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
    category_desc = CATEGORY_DESCRIPTIONS[dp['category']]
    
    return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant. You first think about the reasoning process out loud and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

The following scrambled characters make up the letters of {category_desc} (like an anagram): "{scrambled_word}". Spaces have been added between the scrambled letters for improved legibility, but any spaces in the original name have been removed. Show your reasoning in <think> </think> tags. Once you have thought about it, put your answer between <answer> and </answer> tags.<|eot_id|>'''

if __name__ == '__main__':
    print("Starting anagram generation script...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--names_dir', default='../names', help='Directory containing category txt files')
    parser.add_argument('--local_dir', default='~/data/anagrams')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--total_train_size', type=int, default=100000)
    parser.add_argument('--total_test_size', type=int, default=1024)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    names_dir = Path(args.names_dir)
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # First, count total words to distribute samples proportionally
    total_words = 0
    category_word_counts = {}
    print("\nScanning category files...")
    
    for category_file in names_dir.glob('*.txt'):
        with open(category_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines() 
                    if len(line.strip().replace(" ", "")) >= 3 
                    and is_scrambleable(line.strip())]
            category_word_counts[category_file.name] = len(words)
            total_words += len(words)
            print(f"{category_file.name}: {len(words)} valid words")

    # Calculate proportional samples for each category
    print("\nCalculating category sizes...")
    category_train_sizes = {
        category: max(100, int((count / total_words) * args.total_train_size))
        for category, count in category_word_counts.items()
    }
    category_test_sizes = {
        category: max(10, int((count / total_words) * args.total_test_size))
        for category, count in category_word_counts.items()
    }

    all_train_datasets = []
    all_test_datasets = []

    # Process each category file
    for category_file in names_dir.glob('*.txt'):
        print(f"\nProcessing category: {category_file.name}")
        
        # Load words from file and filter by length and scrambleability
        with open(category_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines() 
                    if len(line.strip().replace(" ", "")) >= 3 
                    and is_scrambleable(line.strip())]
            print(f"Found {len(words)} valid words")

        # Split words into training and test sets
        word_splits = split_words_for_test(
            words,
            test_ratio=args.test_ratio,
            seed_value=args.seed
        )

        # Generate datasets with proportional sizes
        train_samples = gen_dataset(
            word_splits,
            num_samples=category_train_sizes[category_file.name],
            category_name=category_file.name,
            seed_value=args.seed
        )
        
        test_samples = gen_dataset(
            {'train': word_splits['test']},  # Use test words for test set
            num_samples=category_test_sizes[category_file.name],
            category_name=f"{category_file.name} (test)",
            seed_value=args.seed
        )

        # Convert to datasets format
        train_dict = {
            'scrambled_word': [sample[0] for sample in train_samples],
            'target_word': [sample[1] for sample in train_samples],
            'category': [category_file.name] * len(train_samples)
        }
        
        test_dict = {
            'scrambled_word': [sample[0] for sample in test_samples],
            'target_word': [sample[1] for sample in test_samples],
            'category': [category_file.name] * len(test_samples)
        }
        
        def make_map_fn(split):
            def process_fn(example, idx):
                question = make_prefix(example)
                solution = {
                    "scrambled_word": example['scrambled_word'],
                    "target_word": example['target_word'],
                    "category": example['category'],
                    "all_words": words
                }
                data = {
                    "data_source": f"anagram_{category_file.stem}",
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

        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)

        # Print per-category statistics
        print(f"\nStatistics for {category_file.name}:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    # Combine all datasets
    final_train_dataset = concatenate_datasets(all_train_datasets)
    final_test_dataset = concatenate_datasets(all_test_datasets)

    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print(f"Total training samples: {len(final_train_dataset)}")
    print(f"Total test samples: {len(final_test_dataset)}")

    # Save combined datasets
    final_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    final_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # HDFS support
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")