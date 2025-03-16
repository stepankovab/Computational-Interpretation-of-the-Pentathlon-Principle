import re
import eng_to_ipa
from helper.phon_czech import ipa_czech

czech_exceptions = [
        # osm
        (r'\bosm(náct|desát|krát|\b)', r'osum\1'),
        # sedm
        (r'\bsedm(náct|desát|krát|\b)', r'sedum\1'),
        # neučit, neumět
        (r'(\bn[aeiouyáéíóúý])([aeiouyáéíóúý][bcčdďfghjklmnňpqrřsštťvwxzž])',  r'\1 \2')
        ]


def syllabify(text : str, lang : str = 'en', language_specific_letters = 'áčďéěíňóřšťúůýžäöü', language_specific_symbols = "'"):
    """
    TODO
    
    """
    words = re.findall(f"[a-zA-Z{language_specific_letters}{language_specific_letters.upper()}{language_specific_symbols}]+", re.sub(r"[´’]", "'", text))
    syllables = []

    i = 0
    while i < len(words):
        word = words[i].lower()

        # Transcribe to ipa
        if lang == 'en':
            ipa_word = re.sub(r"[ˈ'*]", '', eng_to_ipa.convert(text=word, stress_marks='primary'))
        elif lang == 'cs':
            # Czech-specific, deals with syllabification anomalies
            for (original, replacement) in czech_exceptions:
                word = re.sub(original, replacement, word)
            # Czech-specific, solves problem with prepositions that are written separately but are pronounced as part of the following word
            if (word in ['k', 'v', 's', 'z']) and i < len(words) - 1 and len(words[i + 1]) > 1:
                i += 1
                word = word + words[i]

            ipa_word = re.sub(r"(\bʔ)|'", '', re.sub(r't͡', 't', ipa_czech(word).replace(" ", "")))
        else:
            raise Exception(f"Unknown language: {lang}. Current options are 'cs' for Czech and 'en' for English.")

        letter_counter = 0

        # Get syllables: mask the word and split the mask
        for syllable_mask in _split_mask(_create_word_mask(ipa_word)):
            word_syllable = ""
            for _ in syllable_mask:
                word_syllable += ipa_word[letter_counter]
                letter_counter += 1
                
            syllables.append(word_syllable)
        i += 1

    return syllables


  
def _create_word_mask(word : str) -> str:
    """
    TODO
    """
    word = word.lower()

    vocals = r"[aeiouyæáéíóúýāɑɔəɛɪʊː]"
    consonants = r"[bcdfghjkmnpqstvwxzðňŋɟɡɣɦʔɲʃʒʤʧθř]"
    czech_grey_zone = r"[rl]"

    replacement_rules = [
        # solves 'every' problem
        ('vər', 'v0r'),
        # solves - əling problem
        (r'ə(lɪŋ)', r'0\1'),
        
        # Double consonants
        (r'(' + czech_grey_zone + r')(' + czech_grey_zone + r')', r'\g<1>0'),
        (r'(' + consonants + r')(' + consonants + r')\b', r'\g<1>0'),

        # vowels that are pronounced separately
        (r'[ao][uʊ][iɪ]', 'VCV'),
        (r'[ui][əuɪae]', 'VV'),
        ('ɪɔ', 'VV'),
        ('ɔa', 'VV'),

        # vowels that are pronounced together
        (r'([aeouæáéóúāɑɔəɛʊː])([iyíɪ])([^lr]|$)', r'\1j\3'),
        (vocals + r"{2}", '0V'),
		
        # now all vocals
		(vocals, 'V'),

        # r,l that act like vocals in syllables
		(r'([^V])(' + czech_grey_zone + ')(0*[^0V' + czech_grey_zone[1:-1] + ']|$)', r'\1V\3'),

        # sp, st, sk, ʃt, Cř, Cl, Cr, Cv
		(r's[pt]', 's0'),
		(r'([^V0lr]0*)[řlrv]', r'\g<1>0'),
		(r'([^V0]0*)sk', r'\1s0'),
		(r'([^V0]0*)ʃt', r'\1ʃ0'),

        # finally all consonants
        (czech_grey_zone, 'C'),
		(consonants, 'C')
	]

    for (original, replacement) in replacement_rules:
        word = re.sub(original, replacement, word)

    return word


def _split_mask(mask : str) -> list[str]:
    """
    TODO
    """
    replacements = [
		# vocal at the beginning
		(r'(^0*V)(C0*V)', r'\1/\2'),
		(r'(^0*V0*C0*)C', r'\1/C'),

		# dividing the middle of the word
		(r'(C0*V(C0*$)?)', r'\1/'),
		(r'/(C0*)C', r'\1/C'),
		(r'/(0*V)(0*C0*V)', r'/\1/\2'),
		(r'/(0*V0*C0*)C', r'/\1/C'),

		# add the last consonant to the previous syllable
		(r'/(C0*)$', r'\1/')
	]

    for (original, replacement) in replacements:
        mask = re.sub(original, replacement, mask)

    if len(mask) > 0 and mask[-1] == "/":
        mask = mask[0:-1]

    return mask.split("/")
