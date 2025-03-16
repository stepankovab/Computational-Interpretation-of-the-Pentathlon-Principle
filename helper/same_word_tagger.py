from rhymetagger import RhymeTagger
import re

class SameWordRhymeTagger():
    def __init__(self, lang = "cs") -> None:
        self.lang = lang
        self.rt = RhymeTagger()
        self.rt.load_model(self.lang, verbose=False)

    def load_model(self, lang = "cs"):
        if lang != self.lang:
            self.lang = lang
            self.rt.load_model(self.lang, verbose=False)

    def tag(self, poem, output_format = 3):
        scheme = self.rt.tag(poem=poem, output_format=output_format)
        scheme = self._fill_in_none_rhymes(scheme)

        endings = []
        for line in poem:
            if not line.strip():
                endings.append("")
                continue

            last_word = ''.join([x for x in re.split(r"\s", line.strip())[-1] if x.isalpha])
            endings.append(last_word)

        assert len(scheme) == len(endings)

        for i in range(len(scheme)):
            for j in range(i + 1, len(scheme)):
                if endings[i] == endings[j] and scheme[i] != scheme[j]:
                    scheme[j] = scheme[i] 

        return scheme

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
