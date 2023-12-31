from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PretokenizationArguments:
    """
    Configuration for data pretokenization.
    """

    tokenizer_dir: Optional[str] = field(
        default="gpt2", metadata={"help": "Name or path to the tokenizer."}
    )
    dataset_name: Optional[str] = field(
        default="sedthh/gutenberg_english", metadata={"help": "Name or path to the dataset to pretokenize."}
    )
    tokenized_data_repo: Optional[str] = field(
        default="gutenberg-english-train", metadata={"help": "Repo name of the pretokenized data."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})

@dataclass
class InitializationArguments:
    """
    Configuration for initializing new model.
    """

    config_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Configuration to use for model initialization."}
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Tokenizer attached to model."}
    )
    model_name: Optional[str] = field(default="shakespeare-gpt2", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})

@dataclass
class TrainingArguments:
    """
    Configuration for training model.
    """

    model_ckpt: Optional[str] = field(
        default="HangenYuu/shakespeare-gpt2", metadata={"help": "Model name or path of model to be trained."}
    )
    save_dir: Optional[str] = field(
        default="./", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    dataset_name_train: Optional[str] = field(
        default="HangenYuu/gutenberg-english-train", metadata={"help": "Name or path of training dataset."}
    )
    dataset_name_valid: Optional[str] = field(
        default="HangenYuu/gutenberg-english-train", metadata={"help": "Name or path of validation dataset."}
    )
    train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=10000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=50000, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=1024,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})