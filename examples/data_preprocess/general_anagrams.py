"""
Preprocess dataset for anagram tasks - given scrambled words from various categories,
unscramble them to find the original word. Ensures test set contains some unique words
not seen in training for each category.
"""

import re
import os
from datasets import Dataset
from random import shuffle, seed, choice, sample
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path
import argparse

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

def scramble_word(word: str) -> str:
    """Scramble a word while ensuring it's different from the original.
    Spaces are removed from input, but output has spaces between characters."""
    word = word.lower().strip()
    word_no_spaces = word.replace(" ", "")
    word_list = list(word_no_spaces)
    while True:
        shuffle(word_list)
        scrambled = ' '.join(word_list)  # Add spaces between characters
        if scrambled.replace(" ", "") != word_no_spaces:
            return scrambled

def split_words_for_test(
    words: List[str],
    test_ratio: float = 0.2,
    seed_value: int = 42
) -> Dict[str, List[str]]:
    """Split words into completely separate training and test sets.
    
    Args:
        words: List of words for a specific category
        test_ratio: Proportion of words to use for testing (default 0.2 = 20%)
        seed_value: Random seed for reproducibility
    
    Returns:
        Dictionary containing separate train and test word lists
    """
    seed(seed_value)
    
    # Calculate number of words for test set
    num_test_words = int(len(words) * test_ratio)
    
    # Randomly select words for test set
    test_words = sample(words, num_test_words)
    
    # Remaining words go to train set
    train_words = [word for word in words if word not in test_words]
    
    return {
        'train': train_words,
        'test': test_words
    }

def make_prefix(dp, category_name: str) -> str:
    """Generate the prompt prefix based on the category."""
    scrambled_word = dp['scrambled_word']
    category_desc = CATEGORY_DESCRIPTIONS[category_name]
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant. You first think about the reasoning process out loud and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

The following scrambled characters make up the letters of {category_desc} (like an anagram): "{scrambled_word}". Spaces have been added between the scrambled letters for improved legibility, but any spaces in the original name have been removed. Show your reasoning in <think> </think> tags. Once you have thought about it, put your answer between <answer> and </answer> tags.<|eot_id|>"""

def gen_dataset(
    word_splits: Dict[str, List[str]],
    num_train_samples: int,
    num_test_samples: int,
    seed_value: int = 42,
) -> Dict[str, List[Tuple]]:
    """Generate dataset for anagram task with completely separate train/test words.
    
    Args:
        word_splits: Dictionary containing separate 'train' and 'test' word lists
        num_train_samples: Number of training samples to generate
        num_test_samples: Number of test samples to generate
        seed_value: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test samples
    """
    seed(seed_value)
    train_samples = []
    test_samples = []
    
    # Generate training samples
    for _ in tqdm(range(num_train_samples), desc="Generating training samples"):
        target_word = choice(word_splits['train'])
        scrambled = scramble_word(target_word)
        train_samples.append((scrambled, target_word))
    
    # Generate test samples
    for _ in tqdm(range(num_test_samples), desc="Generating test samples"):
        target_word = choice(word_splits['test'])
        scrambled = scramble_word(target_word)
        test_samples.append((scrambled, target_word))
    
    return {
        'train': train_samples,
        'test': test_samples
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--names_dir', default='../names', help='Directory containing category txt files')
    parser.add_argument('--output_dir', default='./data/anagrams')
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    names_dir = Path(args.names_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each category file
    for category_file in names_dir.glob('*.txt'):
        print(f"\nProcessing category: {category_file.name}")
        
        # Load words from file
        with open(category_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines()]

        # Split words into training and test sets
        word_splits = split_words_for_test(
            words,
            test_ratio=args.test_ratio,
            seed_value=args.seed
        )

        # Generate datasets
        datasets = gen_dataset(
            word_splits,
            num_train_samples=args.train_size,
            num_test_samples=args.test_size,
            seed_value=args.seed
        )

        # Convert to datasets format
        train_dict = {
            'scrambled_word': [sample[0] for sample in datasets['train']],
            'target_word': [sample[1] for sample in datasets['train']]
        }
        
        test_dict = {
            'scrambled_word': [sample[0] for sample in datasets['test']],
            'target_word': [sample[1] for sample in datasets['test']]
        }
        
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        # Add prompts and metadata
        def make_map_fn(split, category):
            def process_fn(example, idx):
                question = make_prefix(example, category)
                solution = {
                    "scrambled_word": example['scrambled_word'],
                    "target_word": example['target_word'],
                    "category": category,
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

        train_dataset = train_dataset.map(
            function=make_map_fn('train', category_file.name),
            with_indices=True
        )
        test_dataset = test_dataset.map(
            function=make_map_fn('test', category_file.name),
            with_indices=True
        )

        # Print statistics
        unique_train_words = set(train_dict['target_word'])
        unique_test_words = set(test_dict['target_word'])
        test_only_words = unique_test_words - unique_train_words
        
        print(f"\nDataset Statistics for {category_file.name}:")
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Total test samples: {len(test_dataset)}")
        print(f"Unique words in training: {len(unique_train_words)}")
        print(f"Unique words in test: {len(unique_test_words)}")
        print(f"Words that appear only in test: {len(test_only_words)}")

        # Save datasets
        category_output_dir = output_dir / category_file.stem
        category_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset.to_parquet(category_output_dir / 'train.parquet')
        test_dataset.to_parquet(category_output_dir / 'test.parquet')