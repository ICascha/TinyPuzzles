import re
import random
from Levenshtein import distance

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before any known prefixes
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("assistant<|end_header_id|>", 1)[1]
    elif "[/INST]<think>" in solution_str:
        solution_str = solution_str.split("[/INST]<think>", 1)[1]
    else:
        return None

    # Get the last line of the response
    solution_str = solution_str.split('\n')[-1]
    
    # Extract content between answer tags
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def normalize_word(word):
    """Normalize word for comparison by removing spaces and converting to lowercase."""
    if word is None:
        return None
    return ''.join(word.lower().split())

def is_valid_word(guess, word_list):
    """Check if the guessed word is in the list of valid words for this category."""
    if guess is None:
        return False
    normalized_guess = normalize_word(guess)
    normalized_words = [normalize_word(word) for word in word_list]
    return normalized_guess in normalized_words

def is_valid_anagram(guess, scrambled_word):
    """Check if the guess is a valid anagram of the scrambled word."""
    if guess is None or scrambled_word is None:
        return False
    normalized_guess = normalize_word(guess)
    normalized_scrambled = normalize_word(scrambled_word)
    return sorted(normalized_guess) == sorted(normalized_scrambled)

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """Score the anagram solution.
    
    Args:
        solution_str: the solution text containing the guessed word
        ground_truth: dictionary containing scrambled_word, target_word, and list of valid words
        format_score: score for guessing any valid word (default 0.1)
        score: score for correct answer (default 1.0)
    
    Returns:
        float: Score based on the correctness of the answer
    """
    scrambled_word = ground_truth['scrambled_word']
    target_word = ground_truth['target_word']
    all_words = ground_truth['all_words']  # List of valid words for this category
    
    # Extract the guessed answer
    guess = extract_solution(solution_str=solution_str)

    do_print = random.randint(0, 20) == 1
    
    if do_print: 
        print(f"--------------------------------")
        print(f"Scrambled word: {scrambled_word}")
        print(f"Target word: {target_word}")
        print(f"Extracted guess: {guess}")
        print(f"Solution string: {solution_str}")
    
    if guess is None:
        return 0
    
    # Normalize the guess and target for comparison
    normalized_guess = normalize_word(guess)
    normalized_target = normalize_word(target_word)
    
    if normalized_guess == normalized_target:
        return score
    return 0  # hard scoring - only exact matches count

def levenshtein_ratio(s1: str, s2: str, distance_val: float) -> float:
    """Calculate the similarity ratio between two strings."""
    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 1.0
    return 1 - (distance_val / max_length)

def compute_metrics(solution_str, ground_truth):
    """Compute metrics for the anagram solution.
    
    Args:
        solution_str: the solution text containing the guessed word
        ground_truth: dictionary containing scrambled_word, target_word, and word list
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    scrambled_word = ground_truth['scrambled_word']
    target_word = ground_truth['target_word']
    all_words = ground_truth['all_words']
    category = ground_truth.get('category', 'unknown')  # Get category if available
    
    # Extract the guessed answer
    guess = extract_solution(solution_str=solution_str)

    if guess is None:
        guess = ""
    
    # Normalize the guess and target for comparison
    normalized_guess = normalize_word(guess)
    normalized_target = normalize_word(target_word)
    
    # Compute various metrics
    valid_word = is_valid_word(guess, all_words)
    valid_anagram = is_valid_anagram(guess, scrambled_word)
    correct_word = normalized_guess == normalized_target
    guess_length = len(normalized_guess) if normalized_guess else 0
    target_length = len(normalized_target) if normalized_target else 0
    distance_val = distance(normalized_guess, normalized_target)
    distance_ratio = levenshtein_ratio(normalized_guess, normalized_target, distance_val)
    length_difference = abs(guess_length - target_length)
    relative_length_difference = length_difference / target_length if target_length > 0 else 1.0
    
    metrics = {
        'valid_word': float(valid_word),
        'valid_anagram': float(valid_anagram),
        'correct_word': float(correct_word),
        'guess_length': float(guess_length),
        'target_length': float(target_length),
        'distance': float(distance_val),
        'distance_ratio': float(distance_ratio),
        'length_difference': float(length_difference),
        'relative_length_difference': float(relative_length_difference),
        'guess': guess,
        'target': target_word,
        'scrambled_word': scrambled_word,
        'category': category,
        'data_source': f"anagram_{category}" if category != 'unknown' else 'unknown'
    }
    
    return metrics