from enum import Enum

class PromptType(Enum):
    baseline = 0
    syllables = 1
    syllables_ends = 2
    keywords = 3
    keywords_ends = 4
    syllables_keywords = 5
    syllables_keywords_ends = 6
    syllables_unrhymed = 7
    syllables_unrhymed_ends = 8
    rhymes = 11
    syllables_rhymes = 12
    syllables_keywords_rhymes = 13
