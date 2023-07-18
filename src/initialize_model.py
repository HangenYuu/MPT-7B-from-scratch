from arguments import InitializationArguments

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load pretrained tokenizer and random initialized model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
tokenizer.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)

# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {
    "scale_attn_by_inverse_layer_idx": True,
    "reorder_and_upcast_attn": True,
}
config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
model = AutoModelForCausalLM.from_config(config)
model.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)