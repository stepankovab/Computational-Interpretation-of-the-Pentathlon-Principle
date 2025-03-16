from helper.IPA_syllabator import syllabify
from helper.same_word_tagger import SameWordRhymeTagger
from helper.rhymer_types import RhymerType
from helper.translate import lindat_translate
from keybert import KeyBERT
from rhymetagger import RhymeTagger

class SectionStructure:
    original_lyrics_list = []
    syllabified_lyrics_list = []
    syllables = []
    rhyme_scheme = []
    keywords = []
    en_keywords = []
    line_keywords = []
    en_line_keywords = []
    num_lines = []

    def __init__(self,
                 original_lyrics_list = [], 
                 syllabified_lyrics_list = [],
                 syllables = [],
                 rhyme_scheme = [],
                 keywords = [],
                 en_keywords = [],
                 line_keywords = [],
                 en_line_keywords = [],
                 num_lines = []):
        
        self.original_lyrics_list = original_lyrics_list
        self.syllabified_lyrics_list = syllabified_lyrics_list
        self.syllables = syllables
        self.rhyme_scheme = rhyme_scheme
        self.keywords = keywords
        self.en_keywords = en_keywords
        self.line_keywords = line_keywords
        self.en_line_keywords = en_line_keywords
        self.num_lines = num_lines

    def to_dict(self):
        structure_dict = {"original_lyrics_list" : self.original_lyrics_list,
                          "syllabified_lyrics_list" : self.syllabified_lyrics_list,
                          "syllables" : self.syllables,
                          "rhyme_scheme" : self.rhyme_scheme,
                          "keywords" : self.keywords,
                          "en_keywords" : self.en_keywords,
                          "line_keywords" : self.line_keywords,
                          "en_line_keywords" : self.en_line_keywords,
                          "num_lines" : self.num_lines}   

        return structure_dict              
    
    def fill_from_dict(self, structure_dict):
        self.original_lyrics_list = structure_dict["original_lyrics_list"]
        self.syllabified_lyrics_list = structure_dict["syllabified_lyrics_list"]
        self.syllables = structure_dict["syllables"]
        self.rhyme_scheme = structure_dict["rhyme_scheme"]
        self.keywords = structure_dict["keywords"]
        self.en_keywords = structure_dict["en_keywords"]
        self.line_keywords = structure_dict["line_keywords"]
        self.en_line_keywords = structure_dict["en_line_keywords"]
        self.num_lines = structure_dict["num_lines"]


class SectionStructureExtractor:
    num_lines: int
    fill_keywords: bool
    fill_line_keywords: bool

    def __init__(self, 
                 kw_model = KeyBERT(), 
                 english_rhyme_detector = SameWordRhymeTagger(),
                 ) -> None:
        
        self.kw_model = kw_model

        if isinstance(english_rhyme_detector, RhymerType):
            if english_rhyme_detector == RhymerType.RHYMETAGGER:
                english_rhyme_detector = RhymeTagger()
            elif english_rhyme_detector == RhymerType.SAME_WORD_RHYMETAGGER:
                english_rhyme_detector = SameWordRhymeTagger("en")

        self.english_rhyme_detector = english_rhyme_detector
        if isinstance(self.english_rhyme_detector, RhymeTagger):
            self.english_rhyme_detector.load_model("en", verbose=False)
        
        if isinstance(self.english_rhyme_detector, SameWordRhymeTagger):
            self.english_rhyme_detector.load_model("en")


    def create_section_structure(self, section, fill_keywords=True, fill_line_keywords=True):
        """
        section: divided by ','
        """
        original_lyrics_list = section.strip().split(",")

        # count lines
        num_lines = len(original_lyrics_list)
        
        # Keywords
        if fill_keywords == True:
            keywords_tuple = self.kw_model.extract_keywords(section)
            en_keywords = [x[0] for x in keywords_tuple]

            cs_keywords_joined = lindat_translate(en_keywords, "en", "cs", ", ")
            keywords = cs_keywords_joined.split(", ")

        # Line keywords
        line_keywords = []
        en_line_keywords = []
        if fill_line_keywords == True:
            for i in range(len(original_lyrics_list)):
                keywords_tuple = self.kw_model.extract_keywords(original_lyrics_list[i])
                if keywords_tuple == []:
                    line_keywords.append("")
                    en_line_keywords.append("")
                    continue
                en_line_keywords.append(' '.join([x[0] for x in keywords_tuple[:min(len(keywords_tuple), 2)]]))

                cs_keywords_line = lindat_translate([en_line_keywords[-1]], "en", "cs", " ")
                line_keywords.append(cs_keywords_line)

        # rhyme scheme
        rhyme_scheme = self.english_rhyme_detector.tag(poem=original_lyrics_list, output_format=3)
        if isinstance(self.english_rhyme_detector, RhymeTagger):
            rhyme_scheme = self._fill_in_none_rhymes(rhyme_scheme)

        # syllables list
        syllabified_lyrics_list = [syllabify(line, 'en') for line in original_lyrics_list]

        # syllables count
        syllables = [len(syll_line) for syll_line in syllabified_lyrics_list]


        return SectionStructure(original_lyrics_list, syllabified_lyrics_list, syllables, rhyme_scheme, keywords, en_keywords, line_keywords, en_line_keywords, num_lines)


    def _fill_in_none_rhymes(self, rhymes):
        """
        Rewrites numeric rhyme scheme into capital letters. Fills in different letters for each None tag.

        Parameters:
        ----------
        rhymes: list of int or None describing the rhyme scheme

        Returns:
        ---------
        rhyme scheme in capital letters
        """
        max_rhyme_ref = 0
        none_ids = []
        for rhyme_i in range(len(rhymes)):
            if isinstance(rhymes[rhyme_i], int):
                if rhymes[rhyme_i] > max_rhyme_ref:
                    max_rhyme_ref = rhymes[rhyme_i]
                # convert to capital letters, start with A
                rhymes[rhyme_i] = chr(64 + rhymes[rhyme_i])
            else:
                none_ids.append(rhyme_i)

        for none_i in none_ids:
            max_rhyme_ref += 1
            rhymes[none_i] = chr(64 + max_rhyme_ref)

        return rhymes

