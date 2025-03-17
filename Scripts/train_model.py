import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from helper.prompt_types import PromptType
from torch.utils.data import DataLoader
import os
from helper.lyrics_datasets import LinesLyricsDataset, WholeLyricsDataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="OSCAR_GPT2", type=str, help="Name of the model. Options: OSCAR_GPT2, BUT_GPT2, TINYLLAMA, BUT_TINYLLAMA") 
parser.add_argument("--model_path", default="./trained_models", type=str, help="Path to the folder with trained models.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--starting_epoch", default=0, type=int, help="epoch to load to continue training, 0 for not loading")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate")
parser.add_argument("--warmup_steps", default=200, type=int, help="Warmup steps")
parser.add_argument("--generation_method", default="whole", type=str, help="whole, lines")
parser.add_argument("--dataset_path", default="./", type=str, help="Path to the training dataset.")
parser.add_argument("--dataset_type", default=0, type=int, help="""Dataset type: 
    BASELINE = 0
    SYLLABLES = 1
    SYLLABLES_ENDS = 2
    WORDS = 3
    WORDS_ENDS = 4
    SYLLABLES_WORDS = 5
    SYLLABLES_WORDS_ENDS = 6
    UNRHYMED_LEN = 7
    UNRHYMED_LEN_END = 8
    RHYME_SCHEME = 11
    SYLLABLES_RHYME_SCHEME = 12
    SYLLABLES_RHYME_SCHEME_WORDS = 13""")
args = parser.parse_args([] if "__file__" not in globals() else None)

try:
    dataset_type = PromptType(args.prompt_type)
except:
    raise ValueError(f"""{args.prompt_type} does not map onto any Dataset type.\n\nBASELINE = 0
    SYLLABLES = 1
    SYLLABLES_ENDS = 2
    WORDS = 3
    WORDS_ENDS = 4
    SYLLABLES_WORDS = 5
    SYLLABLES_WORDS_ENDS = 6
    UNRHYMED_LEN = 7
    UNRHYMED_LEN_END = 8
    RHYME_SCHEME = 11
    SYLLABLES_RHYME_SCHEME = 12
    SYLLABLES_RHYME_SCHEME_WORDS = 13""")


if args.generation_method == "lines":
    dataset = LinesLyricsDataset(lyrics_dataset_path=args.dataset_path, prompt_type=dataset_type)
elif args.generation_method == "whole":
    dataset = WholeLyricsDataset(lyrics_dataset_path=args.dataset_path, dataset_type=dataset_type)
else:
    raise ValueError(f"Unsupported method: {args.generation_method}")

lyrics_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

print(f"----------------- {args.model} ------ {dataset_type.name} ----------------")

device = 'cpu'
if torch.cuda.is_available():
    print("cuda available.")
    device = 'cuda'
    torch.cuda.empty_cache()

if args.model == "OSCAR_GPT2":
    huggingface_path = "lchaloupsky/czech-gpt2-oscar"
elif args.model == "BUT_GPT2":
    huggingface_path = "BUT-FIT/Czech-GPT-2-XL-133k"
elif args.model == "TINYLLAMA":
    huggingface_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
elif args.model == "BUT_TINYLLAMA":
    huggingface_path = "BUT-FIT/CSTinyLlama-1.2B"
else:
    raise ValueError(f"Model {args.model} is not supported for fine-tuning.")

model, tokenizer = AutoModelForCausalLM.from_pretrained(huggingface_path), AutoTokenizer.from_pretrained(huggingface_path)

# Set special tokens if they are not already set
if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
if tokenizer.cls_token is None:
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
model = model.to(device)

if args.starting_epoch != 0:
    model.load_state_dict(state_dict=torch.load(os.path.join(args.model_path, f"{args.model}_{dataset_type.name}_{args.generation_method}_{args.starting_epoch - 1}.pt"), map_location=torch.device(device)))

model.train()
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = args.epochs * (len(lyrics_loader) // args.batch_size))

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)

for epoch in range(args.starting_epoch, args.epochs):
    print('=' * 30 + f" EPOCH {epoch} started " + '=' * 30)
    sum_loss = 0.0
    for idx, batch in enumerate(lyrics_loader):
        for lyrics in batch:
            lyric_tens = torch.tensor(tokenizer.encode(lyrics)).unsqueeze(0).to(device)

            outputs = model(lyric_tens, labels=lyric_tens)
            loss, logits = outputs[:2] 
            sum_loss += loss.detach().data     
            loss.backward()

        optimizer.step()
        scheduler.step() 
        optimizer.zero_grad()
        model.zero_grad()

        if (idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {idx + 1}, Average Loss: {(sum_loss/(idx + 1)):.4f}")

    torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.model}_{args.generation_method}_{dataset_type.name}_{epoch}.pt"))
