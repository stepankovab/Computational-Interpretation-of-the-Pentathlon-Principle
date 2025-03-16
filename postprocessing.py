from pentathlon_principle_evaluator import PentathlonEvaluator
from english_structure_extractor import SectionStructure
from helper.IPA_syllabator import syllabify
from itertools import combinations


class Postprocesser():
    def __init__(self, evaluator = PentathlonEvaluator()) -> None:
        self.evaluator = evaluator
        self.stopwords = {"a","ačkoli","ahoj","ale","anebo","ano","asi","aspoň","během","bez","beze","blízko","bohužel","brzo","bude","budeme","budeš","budete","budou","budu","byl","byla","byli","bylo","byly","bys","čau","chce","chceme","chceš","chcete","chci","chtějí","chtít","chut'","chuti","co","čtrnáct","čtyři","dál","dále","daleko","děkovat","děkujeme","děkuji","den","deset","devatenáct","devět","do","dobrý","docela","dva","dvacet","dvanáct","dvě","hodně","já","jak","jde","je","jeden","jedenáct","jedna","jedno","jednou","jedou","jeho","její","jejich","jemu","jen","jenom","ještě","jestli","jestliže","jí","jich","jím","jimi","jinak","jsem","jsi","jsme","jsou","jste","kam","kde","kdo","kdy","když","kolik","kromě","která","které","kteří","který","kvůli","má","mají","málo","mám","máme","máš","máte","mé","mě","mezi","mí","mít","mně","mffjeez","mnou","moc","mohl","mohou","moje","moji","možná","můj","musí","může","my","na","nad","nade","nám","námi","naproti","nás","náš","naše","naši","ne","ně","nebo","nebyl","nebyla","nebyli","nebyly","něco","nedělá","nedělají","nedělám","neděláme","neděláš","neděláte","nějak","nejsi","někde","někdo","nemají","nemáme","nemáte","neměl","němu","není","nestačí","nevadí","než","nic","nich","ním","nimi","nula","od","ode","on","ona","oni","ono","ony","osm","osmnáct","pak","patnáct","pět","po","pořád","potom","pozdě","před","přes","přese","pro","proč","prosím","prostě","proti","protože","rovně","se","sedm","sedmnáct","šest","šestnáct","skoro","smějí","smí","snad","spolu","sta","sté","sto","ta","tady","tak","takhle","taky","tam","tamhle","tamhleto","tamto","tě","tebe","tebou","ted'","tedy","ten","ti","tisíc","tisíce","to","tobě","tohle","toto","třeba","tři","třináct","trošku","tvá","tvé","tvoje","tvůj","ty","určitě","už","vám","vámi","vás","váš","vaše","vaši","večer","vedle","vlastně","všechno","všichni","vůbec","vy","vždy","za","zač","zatímco","ze","že","aby","aj","ani","az","budem","budes","by","byt","ci","clanek","clanku","clanky","coz","cz","dalsi","design","dnes","email","ho","jako","jej","jeji","jeste","ji","jine","jiz","jses","kdyz","ktera","ktere","kteri","kterou","ktery","ma","mate","mi","mit","muj","muze","nam","napiste","nas","nasi","nejsou","neni","nez","nove","novy","pod","podle","pokud","pouze","prave","pred","pres","pri","proc","proto","protoze","prvni","pta","re","si","strana","sve","svych","svym","svymi","take","takze","tato","tema","tento","teto","tim","timto","tipy","toho","tohoto","tom","tomto","tomuto","tu","tuto","tyto","uz","vam","vas","vase","vice","vsak","zda","zde","zpet","zpravy","a","aniž","až","být","což","či","článek","článku","články","další","i","jenž","jiné","již","jseš","jšte","každý","kteři","ku","me","ná","napište","nechť","ní","nové","nový","o","práve","první","přede","při","sice","své","svůj","svých","svým","svými","také","takže","te","těma","této","tím","tímto","u","více","však","všechen","z","zpět","zprávy"}

    def choose_best_line(self, lines_list, syllables_in = None, ending_in = None, text_in = None, text_in_english=False, remove_add_stopwords=False):
        if len(lines_list) == 0:
            return ""
        
        if text_in != None:
            syllabified_text_in = syllabify(text_in, 'en')
        
        if remove_add_stopwords and syllables_in != None and text_in != None:
            lines_list = self.correct_length_remove_add_stopwords(lines_list, [syllables_in for _ in range(len(lines_list))], [syllabified_text_in for _ in range(len(lines_list))])

        if len(lines_list) == 1:
            return lines_list[0]
        if syllables_in == None and ending_in == None and text_in == None:
            return lines_list[0]
            
        
        singabilities = []
        meanings = []
        perplexities = []
        rhymes = []
        rhythms = []

        # fill in scores dict
        for line_i in range(len(lines_list)):
            sylls = syllabify(lines_list[line_i], 'cs')
            
            if text_in != None:
                singabilities.append(self.evaluator.get_CCVO_distance([sylls], [syllabified_text_in]))
            else:
                singabilities.append(0)
            
            perplexities.append(self.evaluator.compute_perplexity(lines_list[line_i]))

            if text_in != None:
                meanings.append(self.evaluator.get_semantic_similarity(lines_list[line_i], text_in, text1_in_en=False, text2_in_en=text_in_english))
            else:
                meanings.append(0)

            if ending_in != None:
                if len(sylls) == 0:
                    rhyme_score = 0
                else:
                    scheme = self.evaluator.czech_rhyme_detector.tag([ending_in, lines_list[line_i]])
                    if scheme[0] == scheme[1] and scheme[0] != None:
                        rhyme_score = 0
                    else:
                        rhyme_score = 1
                rhymes.append(rhyme_score)
            else:
                rhymes.append(0)

            if syllables_in != None: 
                rhythms.append(self.evaluator.get_section_syllable_distance([syllables_in], [len(sylls)]))
            else:
                rhythms.append(0)

        singabilities = [sorted(singabilities, reverse=False).index(x) for x in singabilities]
        meanings = [sorted(meanings, reverse=True).index(x) for x in meanings]
        perplexities = [sorted(perplexities, reverse=False).index(x) for x in perplexities]
        rhymes = [sorted(rhymes, reverse=False).index(x) * 4 for x in rhymes]
        rhythms = [sorted(rhythms, reverse=False).index(x) for x in rhythms]

        scores = [sum(values) for values in zip(singabilities, meanings, perplexities, rhymes, rhythms)]
        min_line_i = scores.index(min(scores))
        
        return lines_list[min_line_i]



    def choose_best_section(self,
                            lyrics_list,
                            structure : SectionStructure,
                            remove_add_stopwords=False):
        
        if isinstance(lyrics_list, str):
            lyrics_list = [lyrics_list]
            
        if remove_add_stopwords:
            for i in range(len(lyrics_list)):
                lyrics_list[i] = self.correct_length_remove_add_stopwords(lyrics_list[i], structure.syllables, structure.syllabified_lyrics_list)

        if len(lyrics_list) == 0:
            return "", None
        if len(lyrics_list) == 1:
            return lyrics_list[0], None

        scores_dict = self.evaluator.evaluate_outputs_structure([(','.join(lyrics[:min(len(lyrics), structure.num_lines * 2)]), structure) for lyrics in lyrics_list], evaluate_keywords=False, evaluate_line_keywords=False, evaluate_translations=False)        
        score_ranks_dict = {}
        score_ranks_dict["singability"] = [sorted(scores_dict["singability"], reverse=False).index(x) for x in scores_dict["singability"]]
        score_ranks_dict["sense"] = [sorted(scores_dict["sense"], reverse=True).index(x) for x in scores_dict["sense"]]
        score_ranks_dict["naturalness"] = [sorted(scores_dict["naturalness"], reverse=False).index(x) for x in scores_dict["naturalness"]]
        score_ranks_dict["rhyme"] = [sorted(scores_dict["rhyme"], reverse=True).index(x) for x in scores_dict["rhyme"]]
        score_ranks_dict["rhyme2"] = score_ranks_dict["rhyme"].copy()
        score_ranks_dict["rhythm"] = [sorted(scores_dict["rhythm"], reverse=False).index(x) for x in scores_dict["rhythm"]]

        # pick the best match
        scores = [sum(values) for values in zip(*score_ranks_dict.values())]

        ordered_indicies = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]

        for key in scores_dict:
            scores_dict[key] = scores_dict[key][ordered_indicies[0]]

        return lyrics_list[ordered_indicies[0]], scores_dict

    def correct_length_remove_add_stopwords(self, lines, lengths, syllabified_lyrics_list):
        if isinstance(lines, str):
            lines = [lines]

        if len(lines) != len(lengths):
            return lines
        
        for line_i in range(len(lines)):
            
            line = lines[line_i].lower()
            line_len = lengths[line_i]
            difference = len(syllabify(line, 'cs')) - line_len

            if difference == 0 or difference > 2 * line_len:
                continue

            if difference > 0:
                variants = self.generate_variants_without_stopwords(line)

                perplexities = []
                singabilities = []
                rhythms = []
                for variant_i in range(len(variants)):
                    var = variants[variant_i]
                    if line_i != 0:
                        var = lines[line_i - 1] + ', ' + var
                    perplexities.append(self.evaluator.compute_perplexity(var))
                    singabilities.append(self.evaluator.get_CCVO_distance([syllabify(variants[variant_i], 'cs')], [syllabified_lyrics_list[line_i]]))
                    rhythms.append(self.evaluator.get_section_syllable_distance([len(syllabify(variants[variant_i], 'cs'))], [len(syllabified_lyrics_list[line_i])]))
                perplexities = [sorted(perplexities, reverse=False).index(x) for x in perplexities]
                singabilities = [sorted(singabilities, reverse=False).index(x) for x in singabilities]
                rhythms = [sorted(rhythms, reverse=False).index(x) for x in rhythms]
                variants_scores = [sum(values) for values in zip(perplexities, singabilities, rhythms)]
                min_var_i = variants_scores.index(min(variants_scores))
                line = variants[min_var_i]

            if difference < 0 and line.strip():
                possibilities = ["", "tak", "a", "jen", "ten", "ta", "já", "když", "proč", "to", "no", "náš", "nám", "než", "že", "co", "dál", "každý", "že prý", "a tak", "a pak", "copak", "no tak", "tak se ptám", "každý den", "já už vím"]

                perplexities = []
                singabilities = []
                rhythms = []
                for pref_i in range(len(possibilities)):
                    pref = possibilities[pref_i]
                    if line_i != 0:
                        pref = lines[line_i - 1] + ', ' + pref
                    perplexities.append(self.evaluator.compute_perplexity(pref + ' ' + line))
                    singabilities.append(self.evaluator.get_CCVO_distance([syllabify(possibilities[pref_i] + ' ' + line, 'cs')], [syllabified_lyrics_list[line_i]]))
                    rhythms.append(self.evaluator.get_section_syllable_distance([len(syllabify(possibilities[pref_i] + ' ' + line, 'cs'))], [len(syllabified_lyrics_list[line_i])]))
                perplexities = [sorted(perplexities, reverse=False).index(x) for x in perplexities]
                singabilities = [sorted(singabilities, reverse=False).index(x) for x in singabilities]
                rhythms = [sorted(rhythms, reverse=False).index(x) for x in rhythms]
                prefix_scores = [sum(values) for values in zip(perplexities, singabilities, rhythms)]
                min_pref_i = prefix_scores.index(min(prefix_scores))
                prefix = possibilities[min_pref_i]

                if prefix != "":
                    line = prefix + " " + line

            lines[line_i] = line

        return lines
    

    def generate_variants_without_stopwords(self,line):
        words = line.strip().split(" ")

        stopword_indices = [i for i, word in enumerate(words) if word in self.stopwords]
        results = set([" ".join(words)])  # Start with the original sentence

        # Generate combinations of stopwords to remove
        for r in range(1, len(stopword_indices) + 1):
            for indices_to_remove in combinations(stopword_indices, r):
                variant = [word for i, word in enumerate(words[:-1]) if i not in indices_to_remove]
                results.add(" ".join(variant + [words[-1]]))
        
        return list(results)
    