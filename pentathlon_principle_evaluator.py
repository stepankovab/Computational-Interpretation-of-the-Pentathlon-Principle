import Levenshtein as lev
from helper.IPA_syllabator import syllabify
import itertools
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from english_structure_extractor import SectionStructure
from rhymetagger import RhymeTagger
from helper.same_word_tagger import SameWordRhymeTagger
from helper.rhymer_types import RhymerType
from helper.translate import lindat_translate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PentathlonEvaluator():

    def __init__(self, czech_rhyme_detector = SameWordRhymeTagger(lang='cs'), english_rhyme_detector = SameWordRhymeTagger(lang='en'), kw_model = KeyBERT(), embed_model = SentenceTransformer('all-MiniLM-L6-v2'), verbose = False) -> None:
        """
        Parameters
        ---------------
        rt: "cs" RhymeTagger, or any rhymer with 'tag(self, poem : list[str], output_format : int)' method
        kw_model: keyword finder
        embed_model: sentence embedding model
        """
        
        self.kw_model = kw_model
        self.embed_model = embed_model

        LLM_name = "BUT-FIT/csmpt7b"
        llm, tokenizer = AutoModelForCausalLM.from_pretrained(LLM_name), AutoTokenizer.from_pretrained(LLM_name)
        # Set special tokens if they are not already set
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        llm.to(self.device)
        llm.eval()
        self.llm = llm
        self.tokenizer = tokenizer


        if isinstance(czech_rhyme_detector, RhymerType):
            if czech_rhyme_detector == RhymerType.RHYMETAGGER:
                czech_rhyme_detector = RhymeTagger()
                english_rhyme_detector = RhymeTagger()
            elif czech_rhyme_detector == RhymerType.SAME_WORD_RHYMETAGGER:
                czech_rhyme_detector = SameWordRhymeTagger("cs")
                english_rhyme_detector = SameWordRhymeTagger("en")

        self.czech_rhyme_detector = czech_rhyme_detector
        self.english_rhyme_detector = english_rhyme_detector
        self.verbose = verbose

        if isinstance(self.czech_rhyme_detector, RhymeTagger):
            self.czech_rhyme_detector.load_model("cs", verbose=False)
            self.english_rhyme_detector.load_model("en", verbose=False)

        if isinstance(self.czech_rhyme_detector, SameWordRhymeTagger):
            self.czech_rhyme_detector.load_model("cs")
            self.english_rhyme_detector.load_model("en")
    

    def evaluate(self, english_section, czech_section):

        results_dict = {}

        input_list = english_section.split(",")         
        in_syllables = []
        in_syllabified = []

        for line_i in range(len(input_list)):
            line = input_list[line_i]

            if not line.strip():
                in_syllables.append(0)
                in_syllabified.append([])
                continue

            syllabified_line = syllabify(line, 'en')

            if len(syllabified_line) == 0:
                in_syllables.append(0)
                in_syllabified.append([])
                continue

            in_syllables.append(len(syllabified_line))
            in_syllabified.append(syllabified_line)
            

        output_list = czech_section.split(",")         
        out_syllables = []
        out_syllabified = []

        for line_i in range(len(output_list)):
            line = output_list[line_i]

            if not line.strip():
                out_syllables.append(0)
                out_syllabified.append([])
                continue

            syllabified_line = syllabify(line, 'cs')

            if len(syllabified_line) == 0:
                out_syllables.append(0)
                out_syllabified.append([])
                continue

            out_syllables.append(len(syllabified_line))
            out_syllabified.append(syllabified_line)
            
        # Singability: Consonant Cluster Vowel Openness Distance
        results_dict["singability"] = self.get_CCVO_distance(out_syllabified, in_syllabified)

        # Sense: Semantic Similarity
        semantic_similarity = 0
        if len(input_list) > 0:
            semantic_similarity = self.get_semantic_similarity(english_section, czech_section, text2_in_en=False)
        results_dict["sense"] = semantic_similarity

        # Naturalness: Target language model perplexity
        results_dict["naturalness"] = self.compute_perplexity(czech_section)

        # Rhyme: Rhyme Scheme JI
        output_rhyme_scheme = self.czech_rhyme_detector.tag(poem=output_list, output_format=3)
        input_rhyme_scheme = self.english_rhyme_detector.tag(poem=input_list, output_format=3)
        if isinstance(self.czech_rhyme_detector, RhymeTagger):
            output_rhyme_scheme = self._fill_in_none_rhymes(output_rhyme_scheme)
            input_rhyme_scheme = self._fill_in_none_rhymes(input_rhyme_scheme)
        # results_dict["rhyme"].append(self.get_rhyme_scheme_agreement(in_structure.rhyme_scheme, output_rhyme_scheme))
        results_dict["rhyme"] = self.get_rhyme_scheme_jaccard_index(input_rhyme_scheme, output_rhyme_scheme)

        # Rhythm: Syllable Distance
        results_dict["rhythm"] = self.get_section_syllable_distance(in_syllables, out_syllables)

        return results_dict


    def evaluate_outputs_structure(self, outputs_w_structures: list[tuple[str, SectionStructure]]):
        """
        outputs_w_structures: list(str, SectionStructure)
        
        syllables: The expected syllables counts for each line
        endings: The expected line endings for each line
        keywords: The expected keywords of the output
        
        Returns
        ---------------
        results_dict = {"singability" : [],
                        "sense" : [],
                        "naturalness" : [],
                        "rhyme" : [],
                        "rhythm" : [] }
        """
        results_dict = {"singability" : [],
                        "sense" : [],
                        "naturalness" : [],
                        "rhyme" : [],
                        "rhythm" : []
                        }

        for output, in_structure in outputs_w_structures:
            output_list = output.split(",")         
            out_syllables = []
            out_syllabified = []

            for line_i in range(len(output_list)):
                line = output_list[line_i]

                if not line.strip():
                    out_syllables.append(0)
                    out_syllabified.append([])
                    continue

                syllabified_line = syllabify(line, 'cs')

                if len(syllabified_line) == 0:
                    out_syllables.append(0)
                    out_syllabified.append([])
                    continue

                out_syllables.append(len(syllabified_line))
                out_syllabified.append(syllabified_line)

            ##################### metrics ####################

            # Singability: Consonant Cluster Vowel Openness Distance
            results_dict["singability"].append(self.get_CCVO_distance(out_syllabified, in_structure.syllabified_lyrics_list))

            # Sense: Semantic Similarity
            semantic_similarity = 0
            if len(in_structure.original_lyrics_list) > 0:
                semantic_similarity = self.get_semantic_similarity(output, in_structure.original_lyrics_list, text1_in_en=False)
            results_dict["sense"].append(semantic_similarity)

            # Naturalness: Target language model perplexity
            results_dict["naturalness"].append(self.compute_perplexity(output))

            # Rhyme: Rhyme Scheme A.
            output_rhyme_scheme = self.czech_rhyme_detector.tag(poem=output_list, output_format=3)
            if isinstance(self.czech_rhyme_detector, RhymeTagger):
                output_rhyme_scheme = self._fill_in_none_rhymes(output_rhyme_scheme)
            # results_dict["rhyme"].append(self.get_rhyme_scheme_agreement(in_structure.rhyme_scheme, output_rhyme_scheme))
            results_dict["rhyme"].append(self.get_rhyme_scheme_jaccard_index(in_structure.rhyme_scheme, output_rhyme_scheme))

            # Rhythm: Syllable Distance
            results_dict["rhythm"].append(self.get_section_syllable_distance(in_structure.syllables, out_syllables))

        return results_dict
    
    
    def consonant_clusters_vowel_openness(self, syllabified_line):
        """
        TODO
        """
        consonants = r"[bcdfghjkmnpqstvwxzðňŋɟɡɣɦʔɲʃʒʤʧθř]"
        open_vowels = 'aæáāɑ'
        mid_vowels = 'əeoéóɔɛ'
        closed_vowels = 'iuyíúýɪʊ'

        singability_mask = ""
        last_C = 0

        for syllable in syllabified_line:
            syllable = syllable.lower()

            start_C = sum(1 for _ in itertools.takewhile(lambda x: x in consonants, syllable))
            if last_C + start_C >= 3:
                singability_mask += 'C'
            else:
                singability_mask += 'N'


            if any(char in open_vowels for char in syllable):
                singability_mask += 'OO'
            elif any(char in mid_vowels for char in syllable):
                singability_mask += 'VO'
            elif any(char in closed_vowels for char in syllable):
                singability_mask += 'VV'
            else:
                # Czech syllables containing r and l instead of vowels are considered mid, as they are pronouned as the schwa sound
                singability_mask += 'VO'

            last_C = sum(1 for char in syllable[start_C:] if char in consonants)        

        if last_C >= 3:
            singability_mask += 'C'
        else:
            singability_mask += 'N'

        return singability_mask


    def get_CCVO_distance(self, syllabified_sections_list_1, syllabified_sections_list_2):
        """
        TODO
        """
        distance = 0
        
        if len(syllabified_sections_list_1) != len(syllabified_sections_list_2):
            return 10

        for i in range(min(len(syllabified_sections_list_1), len(syllabified_sections_list_2))):

            CCVO_1 = self.consonant_clusters_vowel_openness(syllabified_sections_list_1[i])
            CCVO_2 = self.consonant_clusters_vowel_openness(syllabified_sections_list_2[i])
            
            lev_dist = lev.distance(CCVO_1, CCVO_2)

            distance += (lev_dist / max(min(len(CCVO_1), len(CCVO_2)), 1)) 
            
        distance /= max(min(len(syllabified_sections_list_1), len(syllabified_sections_list_2)), 1)
        
        return distance


    def get_semantic_similarity(self, text1, text2, text1_in_en = True, text2_in_en = True):
        """
        Embed two texts and get their cosine similarity
        """
        if not isinstance(text1, str):
            text1 = ', '.join(text1)
        if not isinstance(text2, str):
            text2 = ', '.join(text2)

        if not text1.strip():
            return 0
        if not text2.strip():
            return 0

        if not text1_in_en and ' '.join(text1).strip():
            text1 = lindat_translate([text1], "cs", "en", " ")

        if not text2_in_en and ' '.join(text2).strip():
            text2 = lindat_translate([text2], "cs", "en", " ")

        embedding1 = self.embed_model.encode(text1, convert_to_tensor=False)
        embedding2 = self.embed_model.encode(text2, convert_to_tensor=False)
        
        cosine_similarity = util.cos_sim(embedding1, embedding2)

        return cosine_similarity[0][0].item()
    

    def compute_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            outputs = self.llm(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = torch.exp(loss)
        return perplexity.item()


    def get_section_syllable_distance(self, syllable_counts_section_1, syllable_counts_section_2):
        """
        TODO
        """
        distance = 0
        
        if len(syllable_counts_section_1) != len(syllable_counts_section_2):
            return 10

        for i in range(min(len(syllable_counts_section_1), len(syllable_counts_section_2))):
            distance += (abs(syllable_counts_section_1[i] - syllable_counts_section_2[i]) / max(syllable_counts_section_1[i], 1)) + (abs(syllable_counts_section_1[i] - syllable_counts_section_2[i]) / max(syllable_counts_section_2[i], 1))
        
        distance /= (2 * min(len(syllable_counts_section_1), len(syllable_counts_section_2)))
        
        return distance


    def get_rhyme_scheme_jaccard_index(self, desired_scheme, new_scheme):
        try:
            assert len(desired_scheme) == len(new_scheme)
            assert len(desired_scheme) > 0
        except:
            return 0

        desired_edges = set()
        new_edges = set()

        for i in range(len(desired_scheme)):
            for j in range(i + 1, len(desired_scheme)):
                if desired_scheme[i] == desired_scheme[j]:
                    desired_edges.add((i,j))
                if new_scheme[i] == new_scheme[j]:
                    new_edges.add((i,j))

        same_edges = desired_edges.intersection(new_edges)
        all_edges = desired_edges.union(new_edges)

        if len(all_edges) == 0:
            return 1

        return len(same_edges) / len(all_edges)
    
    
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
