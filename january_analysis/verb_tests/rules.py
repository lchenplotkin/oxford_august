# rules.py

from typing import Callable, Dict, Any, Set, List


class Rule:
    def __init__(
        self,
        rule_id: str,
        description: str,
        applies_fn: Callable[[Dict[str, Any]], bool],
        check_fn: Callable[[Dict[str, Any]], bool],
        excluded_headwords: Set[str] = None,
    ):
        self.rule_id = rule_id
        self.description = description
        self.applies_fn = applies_fn
        self.check_fn = check_fn
        self.excluded_headwords = excluded_headwords or set()

    def applies(self, token: Dict[str, Any]) -> bool:
        return self.applies_fn(token)

    def check(self, token: Dict[str, Any]) -> bool:
        return self.check_fn(token)

    def is_excluded(self, headword: str) -> bool:
        return headword in self.excluded_headwords


# =========================================================
# Helper lambdas used by multiple rules
# =========================================================

def ends_with(word: str, endings):
    return word.endswith(tuple(endings))


# =========================================================
# RULE DEFINITIONS (ALL your original logic preserved)
# =========================================================

RULES: List[Rule] = []

# RULE 1
RULES.append(Rule(
    rule_id="R1_INF_ENDING",
    description="Infinitive ends in -en or -e unless stem ends in vowel",
    applies_fn=lambda t: t["tag"] == "v%inf",
    check_fn=lambda t: ends_with(t["word"], ["en", "e"]) or t["stem_ends_vowel"],
))

# RULE 3
RULES.append(Rule(
    rule_id="R3_IMP_NO_E",
    description="Imperative singular must not end in -e",
    applies_fn=lambda t: t["tag"].startswith("v%imp"),
    check_fn=lambda t: not t["word"].endswith("e"),
))

# RULE 4
RULES.append(Rule(
    rule_id="R4_PRET_PLURAL_END",
    description="Preterite plural ends in -e or -en",
    applies_fn=lambda t: t["tag"] == "v%pt_pl",
    check_fn=lambda t: ends_with(t["word"], ["e", "en"]),
))

# RULE 5
RULES.append(Rule(
    rule_id="R5_PR_1SG_END",
    description="1st person singular present ends in -e/-en",
    applies_fn=lambda t: t["tag"] == "v%pr_1",
    check_fn=lambda t: ends_with(t["word"], ["e", "en"]),
))

# RULE 6
RULES.append(Rule(
    rule_id="R6_PR_PL_END",
    description="Present plural ends in -e/-en",
    applies_fn=lambda t: t["tag"] == "v%pr_pl",
    check_fn=lambda t: ends_with(t["word"], ["e", "en"]),
))

# RULE 7
RULES.append(Rule(
    rule_id="R7_WEAK_PRET_END",
    description="Weak singular preterite ends in -e/-d/-t",
    applies_fn=lambda t: t["verb_class"] == "weak" and t["tag"] in ["v%pt_1", "v%pt_3"],
    check_fn=lambda t: ends_with(t["word"], ["e", "d", "t"]),
))

# RULE 8
RULES.append(Rule(
    rule_id="R8_STRONG_PPL_END",
    description="Strong past participles end in -e/-en",
    applies_fn=lambda t: t["verb_class"] == "strong" and t["tag"] == "v%ppl",
    check_fn=lambda t: ends_with(t["word"], ["e", "en"]),
))

# RULE 9
RULES.append(Rule(
    rule_id="R9_PRESENT_PART_END",
    description="Present participles end in -e/-en",
    applies_fn=lambda t: t["verb_class"] in ["strong", "weak"] and t["tag"].startswith("v%prp"),
    check_fn=lambda t: ends_with(t["word"], ["e", "en"]),
))

# RULE 10
RULES.append(Rule(
    rule_id="R10_STRONG_PT2_END_E",
    description="2nd person singular strong preterite ends in -e",
    applies_fn=lambda t: t["verb_class"] == "strong" and t["tag"] == "v%pt_2",
    check_fn=lambda t: t["word"].endswith("e"),
))

# RULE 13
RULES.append(Rule(
    rule_id="R13_STRONG_PT1_3_NO_E",
    description="Strong 1st/3rd singular preterite must NOT end in -e",
    applies_fn=lambda t: t["verb_class"] == "strong" and t["tag"] in ["v%pt_1", "v%pt_3"],
    check_fn=lambda t: not t["word"].endswith("e"),
))

# RULE 14
RULES.append(Rule(
    rule_id="R14_WEAK_PT2_NO_E",
    description="Weak 2nd singular preterite must NOT end in -e",
    applies_fn=lambda t: t["verb_class"] == "weak" and t["tag"] == "v%pt_2",
    check_fn=lambda t: not t["word"].endswith("e"),
))

# RULE 15
RULES.append(Rule(
    rule_id="R15_WEAK_PPL_NO_E",
    description="Weak past participle must NOT end in -e",
    applies_fn=lambda t: t["verb_class"] == "weak" and t["tag"] == "v%ppl",
    check_fn=lambda t: not t["word"].endswith("e"),
))

