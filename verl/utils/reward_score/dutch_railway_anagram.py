import re
import random


def extract_solution(solution_str):
    """Extract the guessed station name from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
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
    
    # Check if it's a valid station name
    if not is_valid_station(guess, stations):
        return 0
    
    # Check if it's the correct station
    if normalized_guess == normalized_target:
        return score
    
    # If it's a valid station but not the target, give partial credit
    return format_score
