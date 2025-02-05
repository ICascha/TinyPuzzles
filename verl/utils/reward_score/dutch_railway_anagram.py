import re
import random


def extract_solution(solution_str):
    """Extract the guessed station name from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "assistant<|end_header_id|>" in solution_str:
        # llama-instruct format
        solution_str = solution_str.split("assistant<|end_header_id|>", 1)[1]
    elif "[/INST]<think>" in solution_str:
        # nemo-instruct format
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

def normalize_station_name(station_name):
    """Normalize station name for comparison by removing spaces and converting to lowercase."""
    if station_name is None:
        return None
    return ''.join(station_name.lower().split())

def is_valid_station(guess, stations):
    """Check if the guessed station name is in the list of Dutch railway stations."""
    if guess is None:
        return False
    normalized_guess = normalize_station_name(guess)
    normalized_stations = [normalize_station_name(station) for station in stations]
    return normalized_guess in normalized_stations

def is_valid_anagram(guess, scrambled_word):
    """Check if the guess is a valid anagram of the scrambled word."""
    if guess is None or scrambled_word is None:
        return False
    normalized_guess = normalize_station_name(guess)
    normalized_scrambled = normalize_station_name(scrambled_word)
    return sorted(normalized_guess) == sorted(normalized_scrambled)

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """Score the anagram solution.
    
    Args:
        solution_str: the solution text containing the guessed station name
        ground_truth: dictionary containing scrambled_word and target_station
        format_score: score for guessing any valid station name (default 0.1)
        score: score for correct answer (default 1.0)
    
    Returns:
        float: Score based on the correctness of the answer
    """
    scrambled_word = ground_truth['scrambled_word']
    target_station = ground_truth['target_station']
    stations = ground_truth['stations']  # List of valid Dutch railway station names
    
    # Extract the guessed answer
    guess = extract_solution(solution_str=solution_str)

    do_print = random.randint(0, 20) == 1
    
    if do_print: 
        print(f"--------------------------------")
        print(f"Scrambled word: {scrambled_word}")
        print(f"Target station: {target_station}")
        print(f"Extracted guess: {guess}")
        print(f"Solution string: {solution_str}")
    
    if guess is None:
        return 0
    
    # Normalize the guess and target for comparison
    normalized_guess = normalize_station_name(guess)
    normalized_target = normalize_station_name(target_station)
    
    # First check if it's a valid station name
    if not is_valid_station(guess, stations):
        return 0
    
    # Then check if it's the correct anagram
    if not is_valid_anagram(guess, scrambled_word):
        return format_score
    
    # Finally check if it's the correct station
    if normalized_guess == normalized_target:
        return score
    
    # If it's a valid anagram but wrong station, give partial credit
    return format_score

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate the similarity ratio between two strings."""
    distance = levenshtein_distance(s1, s2)
    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 1.0
    return 1 - (distance / max_length)

def compute_metrics(solution_str, ground_truth):
    """Compute metrics for the anagram solution.
    
    Args:
        solution_str: the solution text containing the guessed station name
        ground_truth: dictionary containing scrambled_word and target_station
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    scrambled_word = ground_truth['scrambled_word']
    target_station = ground_truth['target_station']
    stations = ground_truth['stations']  # List of valid Dutch railway station names
    
    # Extract the guessed answer
    guess = extract_solution(solution_str=solution_str)
    
    # Normalize the guess and target for comparison
    normalized_guess = normalize_station_name(guess)
    normalized_target = normalize_station_name(target_station)    
    # correct station
    valid_station = is_valid_station(guess, stations)
    # correct anagram
    valid_anagram = is_valid_anagram(guess, scrambled_word)
    correct_station = normalized_guess == normalized_target
    # Compute length of the guess
    guess_length = len(normalized_guess)
    # Compute Levenshtein distance
    distance = levenshtein_distance(normalized_guess, normalized_target)
    # Compute normalized ratio
    distance_ratio = levenshtein_ratio(normalized_guess, normalized_target)
    # Compute length difference
    length_difference = abs(len(normalized_guess) - len(normalized_target))
    # Compute relative length difference
    relative_length_difference = length_difference / len(normalized_target)
    
    # Compute metrics
    metrics = {
        'valid_station': valid_station,
        'valid_anagram': valid_anagram,
        'correct_station': correct_station,
        'guess_length': guess_length,
        'distance': distance,
        'distance_ratio': distance_ratio,
        'length_difference': length_difference,
        'relative_length_difference': relative_length_difference
    }
    
    return metrics
