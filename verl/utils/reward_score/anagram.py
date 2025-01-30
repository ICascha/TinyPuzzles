import re
import random


def extract_solution(solution_str):
    """Extract the guessed state name from the solution string."""
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

def normalize_state_name(state_name):
    """Normalize state name for comparison by removing spaces and converting to lowercase."""
    if state_name is None:
        return None
    return ''.join(state_name.lower().split())

def is_valid_state(guess, us_states):
    """Check if the guessed state name is in the list of US states."""
    if guess is None:
        return False
    normalized_guess = normalize_state_name(guess)
    normalized_states = [normalize_state_name(state) for state in us_states]
    return normalized_guess in normalized_states

def is_valid_anagram(guess, scrambled_word):
    """Check if the guess is a valid anagram of the scrambled word."""
    if guess is None or scrambled_word is None:
        return False
    normalized_guess = normalize_state_name(guess)
    normalized_scrambled = normalize_state_name(scrambled_word)
    return sorted(normalized_guess) == sorted(normalized_scrambled)

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """Score the anagram solution.
    
    Args:
        solution_str: the solution text containing the guessed state name
        ground_truth: dictionary containing scrambled_word and target_state
        format_score: score for guessing any valid state name (default 0.1)
        score: score for correct answer (default 1.0)
    
    Returns:
        float: Score based on the correctness of the answer
    """
    scrambled_word = ground_truth['scrambled_word']
    target_state = ground_truth['target_state']
    us_states = ground_truth['us_states']  # List of valid US state names
    
    # Extract the guessed answer
    guess = extract_solution(solution_str=solution_str)

    do_print = random.randint(0, 20) == 1
    
    if do_print: 
        print(f"--------------------------------")
        print(f"Scrambled word: {scrambled_word}")
        print(f"Target state: {target_state}")
        print(f"Extracted guess: {guess}")
        print(f"Solution string: {solution_str}")
    
    if guess is None:
        return 0
    
    # Normalize the guess and target for comparison
    normalized_guess = normalize_state_name(guess)
    normalized_target = normalize_state_name(target_state)
    
    # First check if it's a valid state name
    if not is_valid_state(guess, us_states):
        return 0
    
    # Then check if it's the correct anagram
    if not is_valid_anagram(guess, scrambled_word):
        return format_score
    
    # Finally check if it's the correct state
    if normalized_guess == normalized_target:
        return score
    
    # If it's a valid anagram but wrong state, give partial credit
    return format_score
