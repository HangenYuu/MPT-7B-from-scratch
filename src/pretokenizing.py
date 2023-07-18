# Code modified based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/pretokenizing.py
import time
import multiprocessing

from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from arguments import PretokenizationArguments

def tokenize(example):
    output = {}
    output["input_ids"] = tokenizer(example["TEXT"], truncation=False)["input_ids"]
    output["ratio_char_token"] = len(example["TEXT"]) / len(output["input_ids"]) # type: ignore
    return output

parser = HfArgumentParser(PretokenizationArguments)
args = parser.parse_args()

if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count() // 4

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

t_start = time.time()
ds = load_dataset(args.dataset_name, split="train")
print(f"Dataset loaded in {time.time()-t_start:.2f}s")

t_start = time.time()
ds = ds.map(
    tokenize,
    num_proc=args.num_workers, # type: ignore
    remove_columns=[
        "TEXT",
        "SOURCE",
        "METADATA"
    ],
)
print(f"Dataset tokenized in {time.time()-t_start:.2f}s")

t_start = time.time()
ds.push_to_hub(args.tokenized_data_repo, max_shard_size="300MB")
print(f"Data pushed to the hub in {time.time()-t_start:.2f}s")