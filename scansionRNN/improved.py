import sys
import re
from typing import List, Dict, Tuple, Optional, Set
from itertools import product, combinations

# Special cases for Latin proper nouns where 'u' after vowel is separate syllable
LATIN_PROPER_NOUNS_U_SEPARATE = {
    "theseus": [(4, 5)],  # positions where 'eu' are separate syllables
    # Add more as needed: word -> list of (vowel_idx, u_idx) tuples
}

# Words that can have variable syllable counts (trisyllabic vs disyllabic etc)
VARIABLE_SYLLABLE_WORDS = {
    "troilus": ["troilus", "trolus"],  # Can be tri- or disyllabic
    # Add more as needed
}

# Words with fixed stress patterns
# Format: word -> stress pattern for syllables (e.g., "Su" = stressed then unstressed)
FIXED_STRESS_WORDS = {
    "ladies": "Su",  # First syllable stressed, second unstressed
    "cruel": "Su",
    # Add more as needed
}

# Vowel clusters with fixed stress patterns
# Format: cluster -> stress pattern (e.g., "Su" for two syllables, first stressed)
FIXED_STRESS_CLUSTERS = {
    "ioun": "uS",  # Two syllables: stressed then unstressed
    # Add more as needed
}

ELISION_EXCEPTIONS = ["ye"]

ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
                    "hadden", "his", "her", "him", "hers", "hide", "hir",
                    "hire", "hires", "hirs", "han"]

# Penalty values for constraint satisfaction
PENALTY_NON_E_SILENT = 10      # Marking non-e character as silent
PENALTY_E_BEFORE_VOWEL_UNSTRESSED = 5  # Final e before vowel-initial word marked unstressed instead of silent
PENALTY_E_STRESSED = 1         # Marking any e as stressed
PENALTY_MIDDLE_E_SILENT = 2     # Marking e in middle of word as silent
PENALTY_FINAL_E_STRESSED = 20  # Final e marked stressed
PENALTY_FINAL_E_SILENT = 5
PENALTY_HEADLESS_LINE = 0
PENALTY_FEMININE_ENDING = 0
PENALTY_ENDING_STRESSED = 10
PENALTY_IRREGULAR = 50


def minimal_clean(text: str) -> str:
    """Basic cleaning of text while preserving letters and spaces."""
    retval = re.sub(r"[^a-zA-Z\s]", "", text)
    return retval if not re.match(r"^\s*$", retval) else text

def is_vowel_char(char: str, prev_char: str = None, next_char: str = None, 
                   prev_is_vowel: bool = False) -> bool:
    """
    Determine if i/y acts as a vowel or consonant based on context.
    
    Rules:
    - i/y is vowel if preceded by consonant
    - i/y is consonant if preceded by vowel
    - If i/y is first char and next is consonant -> vowel
    - If i/y is first char and next is vowel -> consonant
    - If i/y appears after i/y -> first is consonant, second is vowel
    - 'qu' counts as consonant
    """
    char_lower = char.lower()
    
    # Standard vowels (excluding i/y for now)
    if char_lower in 'aeou':
        return True
    
    # Handle i/y
    if char_lower in 'iy':
        # First character in word
        if prev_char is None:
            if next_char is None:
                return True
            return next_char.lower() not in 'aeou'  # vowel if next is consonant
        
        prev_lower = prev_char.lower()
        
        # After another i/y -> this one is vowel
        if prev_lower in 'iy':
            return True
        
        # After 'q' -> consonant (part of 'qu' digraph)
        if prev_lower == 'q':
            return False
        
        # After vowel -> consonant
        if prev_lower in 'aeou':
            return False
        
        # After consonant -> vowel
        return True
    
    return False

def get_vowel_positions(word: str) -> List[Tuple[int, str]]:
    """
    Get positions and characters of all vowels in a word.
    Returns list of (position, character) tuples.
    """
    vowels = []
    word_lower = word.lower()
    
    for i, char in enumerate(word_lower):
        prev_char = word_lower[i-1] if i > 0 else None
        next_char = word_lower[i+1] if i < len(word_lower)-1 else None
        
        if char == 'i' and word_lower in ['troilus']:
            vowels.append((i, char))
        elif is_vowel_char(char, prev_char, next_char):
            vowels.append((i, char))
    
    return vowels

def group_vowels_into_syllables(word: str, vowel_positions: List[Tuple[int, str]]) -> List[List[List[Tuple[int, str]]]]:
    """
    Group vowels into syllables based on rules:
    1. Vowels separated by consonants -> separate syllables (deterministic)
    2. 'u' after a vowel -> same syllable (deterministic, with Latin exceptions)
    3. Same vowel character repeated -> same syllable (deterministic)
    4. Adjacent vowels without consonants or grouping -> flexible (can be 1 to N syllables)
    
    Returns list of "vowel groups", where each group contains all possible syllable divisions.
    Each division is a list of syllables, and each syllable is a list of (position, char) tuples.
    
    For deterministic cases, returns single possibility.
    For flexible cases, returns all possibilities from 1 syllable to N syllables.
    """
    if not vowel_positions:
        return []
    
    word_lower = word.lower()
    
    # First, identify vowel clusters (groups separated by consonants)
    clusters = []
    current_cluster = [vowel_positions[0]]
    
    for i in range(1, len(vowel_positions)):
        prev_pos, prev_char = vowel_positions[i-1]
        curr_pos, curr_char = vowel_positions[i]
        
        # Check if there are consonants between the vowels
        has_consonants_between = False
        for j in range(prev_pos + 1, curr_pos):
            char = word_lower[j]
            prev_j = word_lower[j-1] if j > 0 else None
            next_j = word_lower[j+1] if j < len(word_lower)-1 else None
            if not is_vowel_char(char, prev_j, next_j):
                has_consonants_between = True
                break
        
        if has_consonants_between:
            clusters.append(current_cluster)
            current_cluster = [(curr_pos, curr_char)]
        else:
            current_cluster.append((curr_pos, curr_char))
    
    if current_cluster:
        clusters.append(current_cluster)
    
    # Now process each cluster to determine syllable possibilities
    cluster_possibilities = []
    
    for cluster in clusters:
        # Within a cluster, apply u-grouping and same-vowel-grouping rules first
        vowel_groups = []
        current_group = [cluster[0]]
        
        for i in range(1, len(cluster)):
            prev_pos, prev_char = cluster[i-1]
            curr_pos, curr_char = cluster[i]
            
            # Check for Latin proper noun exception
            is_latin_exception = False
            if word_lower in LATIN_PROPER_NOUNS_U_SEPARATE:
                for exception_pair in LATIN_PROPER_NOUNS_U_SEPARATE[word_lower]:
                    if (prev_pos, curr_pos) == exception_pair:
                        is_latin_exception = True
                        break
            
            # Rule 1: 'u' after vowel -> same group (unless Latin exception)
            # Rule 2: Same vowel char repeated -> same group
            if (curr_char == 'u' and not is_latin_exception) or (curr_char == prev_char):
                current_group.append((curr_pos, curr_char))
            else:
                vowel_groups.append(current_group)
                current_group = [(curr_pos, curr_char)]
        
        if current_group:
            vowel_groups.append(current_group)
        
        # Now we have vowel_groups. Each vowel_group is treated as a single unit.
        # Generate all possible syllable divisions from 1 to len(vowel_groups)
        
        if len(vowel_groups) == 1:
            # Only one possibility: all vowels in one syllable
            cluster_possibilities.append([[vowel_groups[0]]])
        else:
            # Generate all partitions from 1 syllable to N syllables
            possibilities = []
            
            # 1 syllable: all vowel_groups together
            possibilities.append([sum(vowel_groups, [])])
            
            # 2 to N syllables: various splits
            for num_syllables in range(2, len(vowel_groups) + 1):
                # Generate all ways to partition vowel_groups into num_syllables parts
                partitions = generate_partitions(vowel_groups, num_syllables)
                possibilities.extend(partitions)
            
            cluster_possibilities.append(possibilities)
    
    return cluster_possibilities

def generate_partitions(items: List[List[Tuple[int, str]]], num_parts: int) -> List[List[List[Tuple[int, str]]]]:
    """
    Generate all ways to partition items into num_parts non-empty parts, maintaining order.
    Each part is a concatenation of consecutive items.
    """
    if num_parts == 1:
        return [[sum(items, [])]]
    
    if num_parts > len(items):
        return []
    
    if num_parts == len(items):
        return [items]
    
    # Use stars and bars approach with order preservation
    # We need to place num_parts-1 dividers among len(items)-1 positions
    
    results = []
    n = len(items)
    
    # Choose positions for dividers (between items)
    for divider_positions in combinations(range(1, n), num_parts - 1):
        partition = []
        prev = 0
        for pos in divider_positions:
            # Concatenate items from prev to pos
            part = sum(items[prev:pos], [])
            partition.append(part)
            prev = pos
        # Add last part
        part = sum(items[prev:], [])
        partition.append(part)
        results.append(partition)
    
    return results

class VowelSyllable:
    """Represents a syllable with its vowel positions and stress information."""
    
    def __init__(self, vowel_positions: List[Tuple[int, str]]):
        self.vowel_positions = vowel_positions
        self.stress = 'u'  # 'S' = stressed, 'u' = unstressed, 'x' = silent/unpronounced
        self.fixed_stress = None  # If set, this syllable has a fixed stress pattern
    
    def get_marked_vowels(self, word: str) -> str:
        """
        Return the vowels with stress marking.
        Uppercase = stressed, lowercase = unstressed, [brackets] = silent
        """
        result = []
        for pos, char in self.vowel_positions:
            if self.stress == 'S':
                result.append(char.upper())
            elif self.stress == 'u':
                result.append(char.lower())
            else:  # silent
                result.append(f"[{char}]")
        return ''.join(result)
    
    def is_silent(self) -> bool:
        return self.stress == 'x'

class WordAnalysis:
    """Represents the syllabic analysis of a word."""
    
    def __init__(self, word: str, syllables: List[VowelSyllable], penalty: int = 0):
        self.word = word
        self.syllables = syllables
        self.penalty = penalty
    
    def get_marked_word(self) -> str:
        """
        Return word with vowel stress markings.
        Example: "aprIlle" for stressed second syllable
        """
        word_chars = list(self.word)
        
        for syllable in self.syllables:
            for pos, orig_char in syllable.vowel_positions:
                if syllable.stress == 'S':
                    word_chars[pos] = orig_char.upper()
                elif syllable.stress == 'u':
                    word_chars[pos] = orig_char.lower()
                else:  # silent
                    # Mark with brackets
                    word_chars[pos] = f"[{orig_char}]"
        
        return ''.join(word_chars)
    
    def get_vowel_pattern(self) -> str:
        """
        Return just the vowel pattern.
        Example: "aIe" for "aprille"
        """
        result = []
        for syllable in self.syllables:
            result.append(syllable.get_marked_vowels(self.word))
        return ' '.join(result)
    
    def get_stress_pattern(self) -> str:
        """
        Return traditional stress pattern (for backward compatibility).
        Example: "uSu"
        """
        pattern = []
        for syllable in self.syllables:
            if not syllable.is_silent():
                pattern.append(syllable.stress)
        return ''.join(pattern)
    
    def get_syllable_count(self) -> int:
        """Return number of pronounced syllables."""
        return sum(1 for s in self.syllables if not s.is_silent())

def analyze_word_syllables(word: str, prev_word: str = None, next_word: str = None) -> List[WordAnalysis]:
    """
    Analyze a word to identify all possible syllable structures.
    Returns list of WordAnalysis objects (one for each possible syllabification).
    """
    word_clean = minimal_clean(word.lower())
    
    # Check if this word has a fixed stress pattern
    if word_clean in FIXED_STRESS_WORDS:
        # Get vowel positions and group into syllables
        vowel_positions = get_vowel_positions(word_clean)
        cluster_possibilities = group_vowels_into_syllables(word_clean, vowel_positions)
        
        # Get the fixed stress pattern
        fixed_pattern = FIXED_STRESS_WORDS[word_clean]
        
        # Find syllabification that matches the required number of syllables
        target_syllable_count = len(fixed_pattern)
        
        for combo in product(*cluster_possibilities):
            syllable_groups = sum(combo, [])
            if len(syllable_groups) == target_syllable_count:
                # Create syllables with fixed stress
                syllables = []
                for i, (group, stress) in enumerate(zip(syllable_groups, fixed_pattern)):
                    syl = VowelSyllable(group)
                    syl.stress = stress
                    syllables.append(syl)
                
                return [WordAnalysis(word_clean, syllables)]
        
        # If we can't match the pattern, fall through to normal analysis
    
    # Get vowel positions
    vowel_positions = get_vowel_positions(word_clean)
    
    # Group into syllables (returns nested structure with all possibilities)
    cluster_possibilities = group_vowels_into_syllables(word_clean, vowel_positions)
    
    # Generate all combinations of cluster possibilities
    all_syllabifications = []
    for combo in product(*cluster_possibilities):
        # Flatten the combination into a single syllabification
        syllable_groups = sum(combo, [])
        all_syllabifications.append(syllable_groups)
    
    # Check if word must have at least one pronounced vowel
    # (single vowel cluster surrounded by consonants on both sides)
    has_initial_consonant = len(word_clean) > 0 and not is_vowel_char(
        word_clean[0], None, word_clean[1] if len(word_clean) > 1 else None)
    has_final_consonant = len(word_clean) > 0 and not is_vowel_char(
        word_clean[-1], 
        word_clean[-2] if len(word_clean) > 1 else None,
        None)
    
    # Check if all vowels are in one cluster (no consonants between them)
    all_in_one_cluster = True
    if len(vowel_positions) > 1:
        for i in range(len(vowel_positions) - 1):
            pos1 = vowel_positions[i][0]
            pos2 = vowel_positions[i + 1][0]
            for j in range(pos1 + 1, pos2):
                char = word_clean[j]
                if not is_vowel_char(char, 
                                    word_clean[j-1] if j > 0 else None,
                                    word_clean[j+1] if j < len(word_clean)-1 else None):
                    all_in_one_cluster = False
                    break
            if not all_in_one_cluster:
                break
    
    must_have_pronounced_vowel = (has_initial_consonant and has_final_consonant and 
                                   all_in_one_cluster and len(vowel_positions) > 0)
    
    # Create WordAnalysis objects for each possible syllabification
    word_analyses = []
    for syllable_groups in all_syllabifications:
        # Create VowelSyllable objects
        syllables = [VowelSyllable(group) for group in syllable_groups]
        
        # Check if any syllables match fixed stress clusters
        for i, syllable in enumerate(syllables):
            # Get the vowel cluster string
            cluster_str = ''.join(char for pos, char in syllable.vowel_positions)
            if cluster_str in FIXED_STRESS_CLUSTERS:
                # This cluster has a fixed stress - need to split it
                fixed_pattern = FIXED_STRESS_CLUSTERS[cluster_str]
                # For now, we'll handle this in the stress assignment phase
                # Mark this syllable for special handling
                syllable.fixed_stress = fixed_pattern
        
        # Apply initial stress rules
        for i, syllable in enumerate(syllables):
            is_final = i == len(syllables) - 1
            
            # Check for final -e
            if is_final and len(syllable.vowel_positions) == 1:
                pos, char = syllable.vowel_positions[0]
                if char == 'e' and pos == len(word_clean) - 1:
                    # Check if preceded by consonant
                    if pos > 0 and not is_vowel_char(word_clean[pos-1], 
                                                      word_clean[pos-2] if pos > 1 else None,
                                                      word_clean[pos]):
                        # Can be silent or unstressed
                        syllable.stress = 'x'  # Default to silent for now
        
        analysis = WordAnalysis(word_clean, syllables)
        analysis.must_have_pronounced_vowel = must_have_pronounced_vowel
        word_analyses.append(analysis)
    
    return word_analyses

def calculate_penalty(word_analysis: WordAnalysis, stress_pattern: List[str], 
                     prev_word: str = None, next_word: str = None) -> int:
    """
    Calculate penalty for a given stress assignment.
    
    Penalties:
    - 10: Non-e vowel marked silent
    - 5: Final e before vowel-initial word marked unstressed (should be silent)
    - 20: Final e marked stressed
    - 2: e in middle of word marked silent
    """
    penalty = 0
    word_lower = word_analysis.word.lower()
    
    next_clean = ''
    # Check if next word starts with vowel
    next_word_vowel_initial = False
    if next_word:
        next_clean = minimal_clean(next_word.lower())
        if next_clean and is_vowel_char(next_clean[0], None, 
                                        next_clean[1] if len(next_clean) > 1 else None):
            next_word_vowel_initial = True
    
    for i, (syllable, stress) in enumerate(zip(word_analysis.syllables, stress_pattern)):
        is_final = i == len(word_analysis.syllables) - 1
        is_middle = not is_final and i > 0
        
        for pos, char in syllable.vowel_positions:
            is_final_char = pos == len(word_lower) - 1
            
            if stress == 'x':  # Silent
                if char != 'e':
                    # Non-e vowel marked silent
                    penalty += PENALTY_NON_E_SILENT
                elif is_middle:
                    # e in middle of word marked silent
                    penalty += PENALTY_MIDDLE_E_SILENT
                elif next_clean and not (is_vowel_char(next_clean[0], None, next_clean[1] if len(next_clean)>1 else None) or next_clean in ELISION_FOLLOWERS) and len(word_analysis.syllables)>1:
                    penalty += PENALTY_FINAL_E_SILENT
            
            if stress == 'S' and char == 'e' and is_final_char and len(word_analysis.syllables)>1:
                penalty += PENALTY_FINAL_E_STRESSED

            if stress == 'S' and char == 'e' and not is_final_char:
                if word_lower[pos+1] in 'rnd' or word_lower.endswith('eth'):
                    penalty += PENALTY_ENDING_STRESSED
                else:
                    penalty += PENALTY_E_STRESSED
            
            if stress == 'u' and char == 'e' and is_final_char and (next_word_vowel_initial or next_clean in ELISION_FOLLOWERS) and word_lower not in ELISION_EXCEPTIONS:
                # Final e before vowel-initial word marked unstressed (should be silent)
                penalty += PENALTY_E_BEFORE_VOWEL_UNSTRESSED
    
    return penalty

def get_all_syllable_possibilities(word_analyses: List[WordAnalysis], 
                                   prev_word: str = None, 
                                   next_word: str = None) -> List[Tuple[WordAnalysis, List[str], int]]:
    """
    Get all possible stress patterns for all syllabifications of a word.
    Returns list of (WordAnalysis, stress_pattern, penalty) tuples.
    
    This handles both syllable structure variation AND stress variation within each structure.
    """
    all_possibilities = []
    
    for word_analysis in word_analyses:
        # Check if this is a fixed stress word (all syllables will have assigned stress already)
        if word_analysis.word in FIXED_STRESS_WORDS:
            # Already has fixed stress, just return it with zero penalty
            stress_pattern = [syl.stress for syl in word_analysis.syllables]
            all_possibilities.append((word_analysis, stress_pattern, 0))
            continue
        
        # Check if word must have at least one pronounced vowel
        must_pronounce = getattr(word_analysis, 'must_have_pronounced_vowel', False)
        
        # For each syllabification, determine possible stress values
        syllable_options = []
        
        for i, syllable in enumerate(word_analysis.syllables):
            is_final = i == len(word_analysis.syllables) - 1
            is_middle = not is_final and i > 0
            
            # Check if this syllable has fixed stress from a vowel cluster
            if hasattr(syllable, 'fixed_stress') and syllable.fixed_stress:
                # This should be handled by splitting the syllable, but for now treat as constraint
                syllable_options.append([syllable.fixed_stress])
                continue
            
            options = []
            
            # Check if this is a final -e
            if is_final and len(syllable.vowel_positions) == 1:
                pos, char = syllable.vowel_positions[0]
                word_lower = word_analysis.word.lower()
                
                if char == 'e' and pos == len(word_lower) - 1 and pos > 0:
                    prev_char = word_lower[pos-1]
                    if not is_vowel_char(prev_char, 
                                        word_lower[pos-2] if pos > 1 else None,
                                        'e'):
                        # Final -e after consonant
                        # Check if next word starts with vowel
                        if next_word:
                            next_clean = minimal_clean(next_word.lower())
                            if next_clean and is_vowel_char(next_clean[0], None, 
                                                           next_clean[1] if len(next_clean) > 1 else None):
                                # Default to silent, allow unstressed and stressed (all with penalties)
                                options = ['x', 'u', 'S']
                            else:
                                # Not before vowel - can be silent, unstressed, or stressed
                                options = ['x', 'u', 'S']
                        else:
                            # No next word - can be silent, unstressed, or stressed
                            next_clean = ''
                            options = ['x', 'u', 'S']
            
            # Check for elision before vowel-initial words (for non-final-e cases)
            if is_final and not options and next_word:
                next_clean = minimal_clean(next_word.lower())
                if next_clean and (next_clean.replace('y','i') in ELISION_FOLLOWERS or is_vowel_char(next_clean[0], None, 
                                               next_clean[1] if len(next_clean) > 1 else None)):
                    # Could be elided
                    options = ['x', 'u', 'S']
            
            # If no special options, syllable can be stressed or unstressed
            if not options:
                options = ['S', 'u']
            
            syllable_options.append(options)
        
        # Generate all stress combinations for this syllabification
        for stress_combo in product(*syllable_options):
            stress_list = list(stress_combo)
            
            # Check constraint: word must have at least one pronounced vowel
            if must_pronounce:
                has_pronounced = any(s != 'x' for s in stress_list)
                if not has_pronounced:
                    continue  # Skip this combination
            
            # Calculate penalty
            penalty = calculate_penalty(word_analysis, stress_list, prev_word, next_word)
            
            all_possibilities.append((word_analysis, stress_list, penalty))
    
    return all_possibilities

def is_pattern_alternating(pattern: str) -> bool:
    """Check if stress pattern is strictly alternating (ignoring spaces and x)."""
    clean_pattern = re.sub(r'[^uS]', '', pattern)
    if len(clean_pattern) <= 1:
        return True
    
    for i in range(1, len(clean_pattern)):
        if clean_pattern[i] == clean_pattern[i-1]:
            return False
    return True

def line_meter_penalty(pattern: str, syllable_count: int, target: int = 10) -> int:
    """
    Apply penalties based on deviation from default line stress pattern.

    Default (0 penalty): uSuSuSuSuS
    +penalty: headless line (SuSuSuSuS)
    +penalty: feminine ending (uSuSuSuSuSu)
    """
    clean = re.sub(r'[^Su]', '', pattern)

    # Normal iambic pentameter
    if clean == "uSuSuSuSuS":
        return 0

    # Headless line (missing initial unstressed syllable)
    elif syllable_count == target - 1:
        if clean == "SuSuSuSuS":
            return PENALTY_HEADLESS_LINE

    # Feminine ending
    elif syllable_count == target + 1:
        if clean == "uSuSuSuSuSu":
            return PENALTY_FEMININE_ENDING
    else:
        return PENALTY_IRREGULAR

    return 0

def scan_line(line: str, target_syllables: int = 10) -> Optional[Tuple[List[WordAnalysis], int]]:
    """
    Perform scansion on a line of text.
    Returns tuple of (list of WordAnalysis objects with stress assignments, total penalty), 
    or None if no valid scansion.
    
    Chooses the scansion with the lowest penalty among all valid alternatives.
    """
    words = minimal_clean(line).split()
    
    # Analyze each word to get all possible syllabifications
    all_word_analyses = []
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else None
        next_word = words[i+1] if i < len(words)-1 else None
        analyses = analyze_word_syllables(word, prev_word, next_word)
        all_word_analyses.append(analyses)
    
    # Get all possible (syllabification, stress, penalty) combinations for each word
    all_possibilities = []
    for i, word_analyses in enumerate(all_word_analyses):
        prev_word = words[i-1] if i > 0 else None
        next_word = words[i+1] if i < len(words)-1 else None
        possibilities = get_all_syllable_possibilities(word_analyses, prev_word, next_word)
        all_possibilities.append(possibilities)
    
    # Try all combinations to find valid scansion with lowest penalty
    best_scansion = None
    best_penalty = float('inf')
    
    for combo in product(*all_possibilities):
        # Each element in combo is (WordAnalysis, stress_list, penalty)
        test_analyses = []
        total_penalty = 0
        
        for word_analysis, stresses, penalty in combo:
            # Create new WordAnalysis with these stresses
            new_syllables = []
            for j, syllable in enumerate(word_analysis.syllables):
                new_syl = VowelSyllable(syllable.vowel_positions)
                new_syl.stress = stresses[j]
                new_syllables.append(new_syl)
            test_analyses.append(WordAnalysis(word_analysis.word, new_syllables))
            total_penalty += penalty
        
        # Count syllables
        total = sum(a.get_syllable_count() for a in test_analyses)
        
        # Build stress pattern
        pattern_parts = []
        for analysis in test_analyses:
            pattern_parts.append(analysis.get_stress_pattern())
        pattern = ' '.join(pattern_parts)
        
        # Check if valid
        valid = False
        if total == target_syllables and is_pattern_alternating(pattern):
            valid = True
        # Allow feminine ending (11 syllables ending in unstressed)
        elif total == target_syllables + 1 and is_pattern_alternating(pattern):
            clean = re.sub(r'[^Su]', '', pattern)
            if clean and clean[-1] == 'u':
                # Check for consonant + en ending
                last_word = words[-1].lower()
                if re.search(r'[^aeou]en$', last_word):
                    valid = True
        
        if valid:
            line_penalty = line_meter_penalty(pattern, total, target_syllables)
            grand_total = total_penalty + line_penalty

            if grand_total < best_penalty:
                best_scansion = test_analyses
                best_penalty = grand_total

    if best_scansion is not None:
        return (best_scansion, best_penalty)
    return None

def format_scansion(word_analyses: List[WordAnalysis]) -> Dict[str, str]:
    """
    Format scansion results in multiple representations.
    """
    marked_words = [a.get_marked_word() for a in word_analyses]
    vowel_patterns = [a.get_vowel_pattern() for a in word_analyses]
    stress_patterns = [a.get_stress_pattern() for a in word_analyses]
    
    return {
        'marked_words': ' '.join(marked_words),
        'vowel_patterns': ' '.join(vowel_patterns),
        'stress_patterns': ' '.join(stress_patterns),
        'syllable_count': sum(a.get_syllable_count() for a in word_analyses)
    }

# Example usage
def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        result = scan_line(line)

        print(line)

        if result:
            word_analyses, penalty = result
            formatted = format_scansion(word_analyses)

            print(formatted["marked_words"])
            print(formatted["stress_patterns"])
            print(f"syllables={formatted['syllable_count']} penalty={penalty}")
        else:
            print("FAILED")

        print()  # blank line between entries


if __name__ == "__main__":
    main()
