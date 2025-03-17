from pentathlon_principle_evaluator import PentathlonEvaluator
from helper.rhymer_types import RhymerType
from english_structure_extractor import SectionStructure
from helper.data_loader import data_loader
from generate_whole import generate_whole
from generate_lines import generate_lines
import time
import json
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="OSCAR_GPT2", type=str, help="Name of the model. Options: OSCAR_GPT2, BUT_GPT2, TINYLLAMA, BUT_TINYLLAMA") 
parser.add_argument("--model_path", default="./trained_models", type=str, help="Path to the state dict of the fine-tuned model.")
parser.add_argument("--input_section", default="let it go,let it go,can't hold it back anymore,let it go,let it go,turn away and slam the door", type=str, help="Input section in English, lines divided by comma ',' and sections divided by semicolon ';' -> eg: let it go,let it go,can't hold it back anymore,let it go,let it go,turn away and slam the door")
parser.add_argument("--prompt_type", default=5, type=int, help="Prompt type the model was fine-tuned on. Submit a number. Options: baseline = 0, syllables = 1, syllables_ends = 2, keywords = 3, keywords_ends = 4, syllables_keywords = 5, syllables_keywords_ends = 6, syllables_unrhymed = 7, syllables_unrhymed_ends = 8, rhymes = 11, syllables_rhymes = 12, syllables_keywords_rhymes = 13")
parser.add_argument("--from_dataset", default=False, type=bool, help="Take test data from Bilingual human-translated lyrics dataset.")
parser.add_argument("--from_structures", default=True, type=str, help="Take test data from the pre-made song structures.")
parser.add_argument("--dataset_path", default="./Data", type=str, help="Path to the test data.")
parser.add_argument("--test_set_size", default=1, type=int, help="Number of samples taken from the test dataset, 0 means all.")
parser.add_argument("--generation_method", default="lines", type=str, help="The method of generating a section the model was finetuned on, or fewshot. Options: whole, lines, fewshot.")
parser.add_argument("--rhymer", default=1, type=int, help="The rhyme detector to be used. Options: SAME_WORD_RHYMETAGGER = 1, RHYMETAGGER = 2")
parser.add_argument("--choose_best", default=10, type=int, help="Choose best postprocessing technique - the number of generated outputs to choose the best from.")
parser.add_argument("--postprocess_stopwords", default=True, type=bool, help="Posrprocess each output by trying to correct the length by removing/adding stopwords.")
parser.add_argument("--results_path", default="./results_dicts", type=str, help="Path to folder to save the results when taking test data from dataset.")
args = parser.parse_args([] if "__file__" not in globals() else None)

########### get input data ###########
if args.from_dataset:
    input_sections = data_loader(args.dataset_path, language="en")
    if args.test_set_size > 0:
        input_sections = input_sections[:min(args.test_set_size, len(input_sections))]

elif args.from_structures:
    with open(os.path.join(args.dataset_path, "new_english_structure_list.json"), "r", encoding='utf-8') as json_file:
        input_sections_dicts_list = json.load(json_file)
    input_sections = []
    if args.test_set_size == 0 or args.test_set_size >= len(input_sections_dicts_list):
        idx = list(range(len(input_sections_dicts_list)))
    else:
        random.seed(13)
        idx = random.sample(list(range(len(input_sections_dicts_list))), min(len(input_sections_dicts_list), args.test_set_size))

    for i in idx:
        section = SectionStructure()
        section.fill_from_dict(input_sections_dicts_list[i])
        input_sections.append(section) 

else:
    input_sections = args.input_section.split(';')

########### generate outputs #############
gen_start_stamp = time.time()
if args.generation_method == "whole":
    result_tripplets = generate_whole(args, input_sections, verbose=False)
elif args.generation_method == "lines":
    result_tripplets = generate_lines(args, input_sections, verbose=False)
gen_end_stamp = time.time()

########### evaluate outputs ##############
eval_start_stamp = time.time()
if None in [trip[2] for trip in result_tripplets]:
    evaluator = PentathlonEvaluator(verbose=False, czech_rhyme_detector=RhymerType(args.rhymer))
    results_dict = evaluator.evaluate_outputs_structure([(t[0], t[1]) for t in result_tripplets], evaluate_keywords=True, evaluate_line_keywords=True, evaluate_translations=True)
else:
    results_dict = {key: [d[key] for d in [trip[2] for trip in result_tripplets]] for key in result_tripplets[0][2].keys()}
eval_stop_stamp = time.time()

########### print results ###############
for lyrics, structure, _ in result_tripplets:
    in_lyrics = '\n'.join(structure.original_lyrics_list)
    out_lyrics = '\n'.join(lyrics.split(','))
    print(f"{'=' * 30}\n\n{in_lyrics}\n\n{out_lyrics}\n")

# print the automatic evaluation of the results
print("=" * 30)
print()
for cat in results_dict:
    if len(results_dict[cat]) == 0:
        continue
    print(f"{cat} -> {round(sum(results_dict[cat]) / len(results_dict[cat]), 2)}")
print()
print("=" * 30)
print()
print(f"eval time: {(eval_stop_stamp - eval_start_stamp) / 60:2f}min")
if len(input_sections) > 0:
    print(f"generation time: {(gen_end_stamp - gen_start_stamp)/60:2f}min, with {((gen_end_stamp - gen_start_stamp)) / len(input_sections):2f}s per lyrics section")

# # save the results into a json file
# if args.from_dataset == True or args.from_structures == True:
lyric_pairs = []
for lyrics, structure, _ in result_tripplets:
    in_lyrics = structure.original_lyrics_list
    out_lyrics = lyrics.split(',')
    lyric_pairs.append((in_lyrics, out_lyrics))
results_dict["lyrics"] = lyric_pairs

if not os.path.exists(args.results_path):
    os.mkdir(args.results_path)

print("==================== dumping results into json =====================")
with open(os.path.join('./', f"new_{args.model}_{args.generation_method}_dataset_type_{args.prompt_type}_samples_{args.test_set_size}_out_per_generation_{args.choose_best}_stopwords_{args.postprocess_stopwords}_rhymer_{args.rhymer}.json"), "w", encoding='utf-8') as json_file:
    json.dump(results_dict, json_file, ensure_ascii=False)

print(f"new_{args.model}_{args.generation_method}_dataset_type_{args.prompt_type}_samples_{args.test_set_size}_out_per_generation_{args.choose_best}_stopwords_{args.postprocess_stopwords}_rhymer_{args.rhymer}.json")
