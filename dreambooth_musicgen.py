#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Fine-tuning MusicGen for text-to-music using 🤗 Transformers Seq2SeqTrainer"""

import logging
import os
import sys

# --- 0. Setup Environment for Kaggle/T4 multi-GPU stability ---
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")

# --- 1. Patch the installed transformers library if needed ---
def patch_transformers():
    try:
        import transformers.models.musicgen.modeling_musicgen as m
        file_path = m.__file__
        with open(file_path, "r") as f:
            content = f.read()

        changed = False
        # Intercept labels in MusicgenForConditionalGeneration.forward to flatten them
        old_labels_check = """        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right("""
                
        new_labels_check = """        # [FIX] Flatten 3D labels to 2D before shift_tokens_right
        if labels is not None and labels.dim() == 3:
            bsz, num_codebooks, seq_len = labels.shape
            labels = labels.reshape(bsz * num_codebooks, seq_len)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right("""

        if "# [FIX] Flatten 3D labels" not in content:
            content = content.replace(old_labels_check, new_labels_check)
            changed = True
        
        # [FIX] Replace .view(-1) with .reshape(-1) for non-contiguous tensors
        if "labels.view(-1)" in content:
            content = content.replace("labels.view(-1)", "labels.reshape(-1)")
            changed = True
        if "logits.view(-1, self.config.vocab_size)" in content:
            content = content.replace("logits.view(-1, self.config.vocab_size)", "logits.reshape(-1, self.config.vocab_size)")
            changed = True

        if changed:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Successfully patched modeling_musicgen: {file_path}")
    except Exception as e:
        # We don't want to crash if we can't patch, but we should log it
        print(f"Warning: Could not patch modeling_musicgen: {e}")

patch_transformers()

# --- 2. Patch MusicgenConfig to have vocab_size for PEFT compatibility ---
try:
    from transformers import MusicgenConfig
    if not hasattr(MusicgenConfig, "vocab_size"):
        MusicgenConfig.vocab_size = 2048
except ImportError:
    pass

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForTextToWaveform,
    AutoModel,
    AutoProcessor,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.integrations import is_wandb_available
from multiprocess import set_start_method


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.40.0.dev0")
require_version("datasets>=2.12.0")


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


#### ARGUMENTS


class MusicgenTrainer(Seq2SeqTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overriding evaluate to ensure eval_loss is always present in metrics,
        especially when compute_metrics is None.
        """
        # If compute_metrics is None, we should force prediction_loss_only to True
        # to ensure the loss is computed and returned as 'eval_loss'.
        original_prediction_loss_only = self.args.prediction_loss_only
        if self.compute_metrics is None:
            self.args.prediction_loss_only = True
        
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            # Restore original state
            self.args.prediction_loss_only = original_prediction_loss_only
            
        return metrics

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length, tensor.shape[2]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        length = min(max_length, tensor.shape[1])
        padded_tensor[:, :length] = tensor[:, :length]
        return padded_tensor


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained processor name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    pad_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model pad token id."},
    )
    decoder_start_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model decoder start token id."},
    )
    freeze_text_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the text encoder."},
    )
    clap_model_name_or_path: str = field(
        default="laion/larger_clap_music_and_speech",
        metadata={
            "help": "Used to compute audio similarity during evaluation. Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use Lora."},
    )
    guidance_scale: float = field(
        default=None,
        metadata={"help": "If specified, change the model guidance scale."},
    )


@dataclass
class DataSeq2SeqTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """

    dataset_name: str = field(
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    target_audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default=None,
        metadata={
            "help": "If set, the name of the description column containing the text data. If not, you should set `add_metadata` to True, to automatically generates music descriptions ."
        },
    )
    instance_prompt: str = field(
        default=None,
        metadata={
            "help": "If set and `add_metadata=True`, will add the instance prompt to the music description. For example, if you set this to `punk`, `punk` will be added to the descriptions. This allows to use this instance prompt as an anchor for your model to learn to associate it to the specificities of your dataset."
        },
    )
    conditional_audio_column_name: str = field(
        default=None,
        metadata={
            "help": "If set, the name of the dataset column containing conditional audio data. This is entirely optional and only used for conditional guided generation."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    full_generation_sample_text: str = field(
        default="80s blues track.",
        metadata={
            "help": (
                "This prompt will be used during evaluation as an additional generated sample."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={
            "help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."
            "Generates audio at the beginning and the end of the training to show evolution."
        },
    )
    add_metadata: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, automatically generates song descriptions, using librosa and msclap."
                "Don't forget to install these libraries: `pip install msclap librosa`"
            )
        },
    )
    push_metadata_repo_id: str = field(
        default=None,
        metadata={
            "help": (
                "if specified and `add_metada=True`, will push the enriched dataset to the hub. Useful if you want to compute it only once."
            )
        },
    )
    num_samples_to_generate: int = field(
        default=4,
        metadata={
            "help": (
                "If logging with `wandb`, indicates the number of samples from the test set to generate"
            )
        },
    )
    audio_separation: bool = field(
        default=False,
        metadata={"help": ("If set, performs audio separation using demucs.")},
    )
    audio_separation_batch_size: int = field(
        default=10,
        metadata={
            "help": (
                "If `audio_separation`, indicates the batch size passed to demucs."
            )
        },
    )


@dataclass
class DataCollatorMusicGenWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        labels = [
            torch.tensor(feature["labels"]).transpose(0, 1) for feature in features
        ]
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        # [FIX] Transpose to [bsz, num_codebooks, seq_len] for DataParallel
        labels = labels.transpose(1, 2).contiguous()

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        input_ids = self.processor.tokenizer.pad(input_ids, return_tensors="pt")

        batch = {"labels": labels, **input_ids}

        if self.feature_extractor_input_name in features[0]:
            input_values = [
                {
                    self.feature_extractor_input_name: feature[
                        self.feature_extractor_input_name
                    ]
                }
                for feature in features
            ]
            input_values = self.processor.feature_extractor.pad(
                input_values, return_tensors="pt"
            )

            batch[self.feature_extractor_input_name] = input_values

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataSeq2SeqTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [FIX] Kaggle specific optimizations for stability and cleaner logs
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        training_args.disable_tqdm = True
        if training_args.eval_accumulation_steps is None or training_args.eval_accumulation_steps <= 0:
            training_args.eval_accumulation_steps = 1
        logger.info("Kaggle environment detected: tqdm disabled, eval_accumulation_steps set to 1.")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_musicgen_melody", model_args, data_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict()
    num_workers = data_args.preprocessing_num_workers
    add_metadata = data_args.add_metadata

    if add_metadata and data_args.text_column_name:
        raise ValueError(
            "add_metadata and text_column_name are both True, chose the former if you want automatically generated music descriptions or the latter if you want to use your own set of descriptions."
        )

    # [FIX] Optimized Loading: Check if it's a pre-processed directory from save_to_disk
    if os.path.isdir(data_args.dataset_name) and (
        os.path.exists(os.path.join(data_args.dataset_name, "dataset_dict.json")) or 
        os.path.exists(os.path.join(data_args.dataset_name, "state.json"))
    ):
        logger.info(f"Loading pre-processed dataset from {data_args.dataset_name}...")
        vectorized_datasets = datasets.load_from_disk(data_args.dataset_name)
        
        # [FIX] Remap 'test' to 'eval' for compatibility with Trainer
        if "test" in vectorized_datasets and "eval" not in vectorized_datasets:
            vectorized_datasets["eval"] = vectorized_datasets["test"]
            
        # [FIX] Aggressive capping immediately after load for Kaggle
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ and training_args.do_eval:
            for split_name in ["test", "eval"]:
                if split_name in vectorized_datasets:
                    if len(vectorized_datasets[split_name]) > 50:
                        logger.warning(f"Kaggle: Capping {split_name} split to 50 samples.")
                        vectorized_datasets[split_name] = vectorized_datasets[split_name].select(range(50))

        # Bypassing all preprocessing steps below
        raw_datasets = None 
    else:
        # Optimized Loading: Check for local JSONL files to avoid audio copy/cache
        data_files = {}
        if os.path.isdir(data_args.dataset_name):
            for split in ["train", "test", "eval"]:
                jsonl_path = os.path.join(data_args.dataset_name, f"{split}.jsonl")
                if os.path.exists(jsonl_path):
                    data_files[split] = jsonl_path
        
        if data_files:
            logger.info(f"Loading dataset from local JSONL files: {data_files}")
            local_raw_datasets = load_dataset("json", data_files=data_files)
            if "train" in local_raw_datasets and training_args.do_train:
                raw_datasets["train"] = local_raw_datasets["train"]
            if "test" in local_raw_datasets and training_args.do_eval:
                raw_datasets["eval"] = local_raw_datasets["test"]
            if "eval" in local_raw_datasets and training_args.do_eval:
                 raw_datasets["eval"] = local_raw_datasets["eval"]
            
            # If testing on Vietnamese dataset, it uses 'audio_path' column
            if "audio_path" in raw_datasets[next(iter(raw_datasets.keys()))].column_names:
                data_args.target_audio_column_name = "audio_path"
        else:
            if training_args.do_train:
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=data_args.train_split_name,
                    num_proc=num_workers,
                )

                if data_args.max_train_samples is not None:
                    raw_datasets["train"] = (
                        raw_datasets["train"]
                        .shuffle()
                        .select(range(data_args.max_train_samples))
                    )

            if training_args.do_eval:
                raw_datasets["eval"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=data_args.eval_split_name,
                    num_proc=num_workers,
                )

                if data_args.max_eval_samples is not None:
                    raw_datasets["eval"] = raw_datasets["eval"].select(
                        range(data_args.max_eval_samples)
                    )

    if raw_datasets:
        # Standard preprocessing logic (only if not loaded from disk)
        if training_args.do_train:
            if data_args.target_audio_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--target_audio_column_name '{data_args.target_audio_column_name}' not found in dataset."
                    " Make sure to set `--target_audio_column_name` to the correct audio column."
                )

            if data_args.instance_prompt is not None:
                logger.warning(
                    f"Using the following instance prompt: {data_args.instance_prompt}"
                )
            elif data_args.text_column_name not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--text_column_name {data_args.text_column_name} not found in dataset. "
                    "Make sure to set `--text_column_name` to the correct text column."
                )
            elif data_args.text_column_name is None and data_args.instance_prompt is None:
                raise ValueError("--instance_prompt or --text_column_name must be set.")

        # [EXISTING PREPROCESSING LOGIC CONTINUES...]

    if data_args.audio_separation:
        try:
            from demucs import pretrained
        except ImportError:
            print(
                "To perform audio separation, you should install additional packages, run: `pip install -e .[metadata]]` or `pip install demucs`."
            )
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        from datasets import Audio

        demucs = pretrained.get_model("htdemucs")
        if torch.cuda.device_count() > 0:
            demucs.to("cuda:0")

        audio_column_name = data_args.target_audio_column_name

        def wrap_audio(audio, sr):
            return {"array": audio.cpu().numpy(), "sampling_rate": sr}

        def filter_stems(batch, rank=None):
            device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
            if rank is not None:
                # move the model to the right GPU if not there already
                device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
                # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
                demucs.to(device)

            if isinstance(batch[audio_column_name], list):
                wavs = [
                    convert_audio(
                        torch.tensor(audio["array"][None], device=device).to(
                            torch.float32
                        ),
                        audio["sampling_rate"],
                        demucs.samplerate,
                        demucs.audio_channels,
                    ).T
                    for audio in batch["audio"]
                ]
                wavs_length = [audio.shape[0] for audio in wavs]

                wavs = torch.nn.utils.rnn.pad_sequence(
                    wavs, batch_first=True, padding_value=0.0
                ).transpose(1, 2)
                stems = apply_model(demucs, wavs)

                batch[audio_column_name] = [
                    wrap_audio(s[:-1, :, :length].sum(0).mean(0), demucs.samplerate)
                    for (s, length) in zip(stems, wavs_length)
                ]

            else:
                audio = torch.tensor(
                    batch[audio_column_name]["array"].squeeze(), device=device
                ).to(torch.float32)
                sample_rate = batch[audio_column_name]["sampling_rate"]
                audio = convert_audio(
                    audio, sample_rate, demucs.samplerate, demucs.audio_channels
                )
                stems = apply_model(demucs, audio[None])

                batch[audio_column_name] = wrap_audio(
                    stems[0, :-1].mean(0), demucs.samplerate
                )

            return batch

        num_proc = (
            torch.cuda.device_count() if torch.cuda.device_count() >= 1 else num_workers
        )

        raw_datasets = raw_datasets.map(
            filter_stems,
            batched=True,
            batch_size=data_args.audio_separation_batch_size,
            with_rank=True,
            num_proc=num_proc,
        )
        raw_datasets = raw_datasets.cast_column(audio_column_name, Audio())

        del demucs

    if add_metadata:
        try:
            from msclap import CLAP
            import librosa
        except ImportError:
            print(
                "To add metadata, you should install additional packages, run: `pip install -e .[metadata]]"
            )
        from utils import instrument_classes, genre_labels, mood_theme_classes
        import tempfile
        import torchaudio
        import random

        clap_model = CLAP(version="2023", use_cuda=False)
        instrument_embeddings = clap_model.get_text_embeddings(instrument_classes)
        genre_embeddings = clap_model.get_text_embeddings(genre_labels)
        mood_embeddings = clap_model.get_text_embeddings(mood_theme_classes)

        def enrich_text(batch):
            audio, sampling_rate = (
                batch["audio"]["array"],
                batch["audio"]["sampling_rate"],
            )

            tempo, _ = librosa.beat.beat_track(y=audio, sr=sampling_rate)
            tempo = f"{str(round(tempo))} bpm"  # not usually accurate lol
            chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
            key = np.argmax(np.sum(chroma, axis=1))
            key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][key]

            with tempfile.TemporaryDirectory() as tempdir:
                path = os.path.join(tempdir, "tmp.wav")
                torchaudio.save(path, torch.tensor(audio).unsqueeze(0), sampling_rate)
                audio_embeddings = clap_model.get_audio_embeddings([path])

            instrument = clap_model.compute_similarity(
                audio_embeddings, instrument_embeddings
            ).argmax(dim=1)[0]
            genre = clap_model.compute_similarity(
                audio_embeddings, genre_embeddings
            ).argmax(dim=1)[0]
            mood = clap_model.compute_similarity(
                audio_embeddings, mood_embeddings
            ).argmax(dim=1)[0]

            instrument = instrument_classes[instrument]
            genre = genre_labels[genre]
            mood = mood_theme_classes[mood]

            metadata = [key, tempo, instrument, genre, mood]

            random.shuffle(metadata)
            batch["metadata"] = ", ".join(metadata)
            return batch

        raw_datasets = raw_datasets.map(
            enrich_text,
            num_proc=1 if torch.cuda.device_count() > 0 else num_workers,
            desc="add metadata",
        )

        del clap_model, instrument_embeddings, genre_embeddings, mood_embeddings

        if data_args.push_metadata_repo_id:
            raw_datasets.push_to_hub(data_args.push_metadata_repo_id)

    # 3. Next, let's load the config as we might need it to create
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        revision=model_args.model_revision,
    )

    # update pad token id and decoder_start_token_id
    config.update(
        {
            "pad_token_id": model_args.pad_token_id
            if model_args.pad_token_id is not None
            else model.config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else model.config.decoder_start_token_id,
        }
    )
    config.decoder.update(
        {
            "pad_token_id": model_args.pad_token_id
            if model_args.pad_token_id is not None
            else model.config.decoder.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else model.config.decoder.decoder_start_token_id,
        }
    )

    # 4. Now we can instantiate the processor and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load processor
    processor = AutoProcessor.from_pretrained(
        model_args.processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    instance_prompt = data_args.instance_prompt
    instance_prompt_tokenized = None
    full_generation_sample_text = data_args.full_generation_sample_text
    if data_args.instance_prompt is not None:
        instance_prompt_tokenized = processor.tokenizer(instance_prompt)
    if full_generation_sample_text is not None:
        full_generation_sample_text = processor.tokenizer(
            full_generation_sample_text, return_tensors="pt"
        )

    # create model
    model = AutoModelForTextToWaveform.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        revision=model_args.model_revision,
    )

    model.config.vocab_size = model.decoder.config.vocab_size
    model.config.pad_token_id = 2047
    model.config.decoder_start_token_id = 2047
    # take audio_encoder_feature_extractor
    audio_encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(
        model.config.audio_encoder._name_or_path,
    )

    # Training constants needed for DataCollator and processing
    feature_extractor_input_name = processor.feature_extractor.model_input_names[0]
    audio_encoder_pad_token_id = config.decoder.pad_token_id
    num_codebooks = model.decoder.config.num_codebooks

    # 5. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    if raw_datasets is not None:
        # resample target audio
        features = next(iter(raw_datasets.values())).features
        audio_feature = features[data_args.target_audio_column_name]

        # [FIX] Only check sampling_rate if it's an Audio feature. 
        # If it's a string path (Zero-Copy), we handle resampling during manual loading.
        if hasattr(audio_feature, "sampling_rate"):
            dataset_sampling_rate = audio_feature.sampling_rate
            if dataset_sampling_rate != audio_encoder_feature_extractor.sampling_rate:
                raw_datasets = raw_datasets.cast_column(
                    data_args.target_audio_column_name,
                    datasets.features.Audio(
                        sampling_rate=audio_encoder_feature_extractor.sampling_rate
                    ),
                )

        if data_args.conditional_audio_column_name is not None:
            cond_audio_feature = features[data_args.conditional_audio_column_name]
            if hasattr(cond_audio_feature, "sampling_rate"):
                dataset_sampling_rate = cond_audio_feature.sampling_rate
                if dataset_sampling_rate != processor.feature_extractor.sampling_rate:
                    raw_datasets = raw_datasets.cast_column(
                        data_args.conditional_audio_column_name,
                        datasets.features.Audio(
                            sampling_rate=processor.feature_extractor.sampling_rate
                        ),
                    )

        # derive max & min input length for sample rate & max duration
        max_target_length = (
            data_args.max_duration_in_seconds
            * audio_encoder_feature_extractor.sampling_rate
        )
        min_target_length = (
            data_args.min_duration_in_seconds
            * audio_encoder_feature_extractor.sampling_rate
        )
        target_audio_column_name = data_args.target_audio_column_name
        conditional_audio_column_name = data_args.conditional_audio_column_name
        text_column_name = data_args.text_column_name

        if data_args.instance_prompt is not None:
            with training_args.main_process_first(desc="instance_prompt preprocessing"):
                # compute text embeddings on one process since it's only a forward pass
                # do it on CPU for simplicity
                instance_prompt_tokenized = instance_prompt_tokenized["input_ids"]

        # Preprocessing the datasets.
        # We need to read the audio files as arrays and tokenize the targets.
        def prepare_audio_features(batch):
            # load audio
            if conditional_audio_column_name is not None:
                sample = batch[conditional_audio_column_name]
                inputs = processor.feature_extractor(
                    sample["array"], sampling_rate=sample["sampling_rate"]
                )
                batch[feature_extractor_input_name] = getattr(
                    inputs, feature_extractor_input_name
                )[0]

            if text_column_name is not None:
                text = batch[text_column_name]
                batch["input_ids"] = processor.tokenizer(text)["input_ids"]
            elif add_metadata is not None and "metadata" in batch:
                metadata = batch["metadata"]
                if instance_prompt is not None and instance_prompt != "":
                    metadata = f"{instance_prompt}, {metadata}"
                batch["input_ids"] = processor.tokenizer(metadata)["input_ids"]
            else:
                batch["input_ids"] = instance_prompt_tokenized

            # load audio
            target_sample = batch[target_audio_column_name]
            
            # [FIX] Manual load if it's a path (string) to avoid HF cache explosion
            if isinstance(target_sample, str):
                import librosa
                audio_array, _ = librosa.load(
                    target_sample, 
                    sr=audio_encoder_feature_extractor.sampling_rate,
                    mono=True
                )
                sampling_rate = audio_encoder_feature_extractor.sampling_rate
            else:
                audio_array = target_sample["array"].squeeze()
                sampling_rate = target_sample["sampling_rate"]

            # [FIX] Truncate audio if it's longer than max_target_length
            if len(audio_array) > max_target_length:
                audio_array = audio_array[:int(max_target_length)]
                
            labels = audio_encoder_feature_extractor(
                audio_array, sampling_rate=sampling_rate
            )
            batch["labels"] = labels["input_values"]

            # take length of raw audio waveform
            batch["target_length"] = len(audio_array)
            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_audio_features,
                remove_columns=next(iter(raw_datasets.values())).column_names,
                num_proc=num_workers,
                desc="preprocess datasets",
            )

            def is_audio_in_length_range(length):
                # [FIX] We now only filter out audio that is TOO SHORT. 
                # Audio that is too long is truncated in prepare_audio_features.
                return length >= min_target_length

            # filter data that is shorter than min_target_length
            vectorized_datasets = vectorized_datasets.filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["target_length"],
            )

        audio_decoder = model.audio_encoder

        pad_labels = torch.ones((1, 1, num_codebooks, 1)) * audio_encoder_pad_token_id

        if torch.cuda.device_count() == 1:
            audio_decoder.to("cuda")

        def apply_audio_decoder(batch, rank=None):
            if rank is not None:
                # move the model to the right GPU if not there already
                device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
                audio_decoder.to(device)

            with torch.no_grad():
                labels = audio_decoder.encode(
                    torch.tensor(batch["labels"]).to(audio_decoder.device)
                )["audio_codes"]

            # add pad token column
            labels = torch.cat(
                [pad_labels.to(labels.device).to(labels.dtype), labels], dim=-1
            )

            labels, delay_pattern_mask = model.decoder.build_delay_pattern_mask(
                labels.squeeze(0),
                audio_encoder_pad_token_id,
                labels.shape[-1] + num_codebooks,
            )

            labels = model.decoder.apply_delay_pattern_mask(labels, delay_pattern_mask)

            # the first timestamp is associated to a row full of BOS, let's get rid of it
            batch["labels"] = labels[:, 1:].cpu()
            return batch

        with training_args.main_process_first(desc="audio target preprocessing"):
            # Encodec doesn't truely support batching
            # Pass samples one by one to the GPU
            vectorized_datasets = vectorized_datasets.map(
                apply_audio_decoder,
                with_rank=True,
                num_proc=torch.cuda.device_count()
                if torch.cuda.device_count() > 0
                else num_workers,
                desc="Apply encodec",
            )

    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:
        if is_wandb_available():
            from transformers.integrations import WandbCallback
        else:
            raise ValueError(
                "`args.add_audio_samples_to_wandb=True` and `wandb` in `report_to` but wandb is not installed. See https://docs.wandb.ai/quickstart to install."
            )

    # 6. Next, we can prepare the training.
    # instantiate a data collator and the trainer

    # [FIX] Removed CLAP similarity computation to save RAM and prevent KeyError: 'eval_loss'
    # For fine-tuning, eval_loss is sufficient and much more stable.
    compute_metrics = None

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            processor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorMusicGenWithPadding(
        processor=processor,
        feature_extractor_input_name=feature_extractor_input_name,
    )

    # Freeze Encoders
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    if model_args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False

    if model_args.guidance_scale is not None:
        model.generation_config.guidance_scale = model_args.guidance_scale

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        # TODO(YL): add modularity here
        target_modules = (
            [
                "enc_to_dec_proj",
                "audio_enc_to_dec_proj",
                "k_proj",
                "v_proj",
                "q_proj",
                "out_proj",
                "fc1",
                "fc2",
                "lm_heads.0",
            ]
            + [f"lm_heads.{str(i)}" for i in range(len(model.decoder.lm_heads))]
            + [f"embed_tokens.{str(i)}" for i in range(len(model.decoder.lm_heads))]
        )

        if not model_args.freeze_text_encoder:
            target_modules.extend(["k", "v", "q", "o", "wi", "wo"])

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        logger.info(f"Modules with Lora: {model.targeted_module_names}")

    # [FIX] Ensure evaluation dataset is present if do_eval is True
    if training_args.do_eval and ("eval" not in vectorized_datasets or len(vectorized_datasets["eval"]) == 0):
        logger.warning("Evaluation split 'eval' not found or empty. Creating a small eval split from 'train'.")
        if "train" in vectorized_datasets and len(vectorized_datasets["train"]) > 1:
            # Split train into train and eval (90/10)
            split_dataset = vectorized_datasets["train"].train_test_split(test_size=0.1, seed=training_args.seed)
            vectorized_datasets["train"] = split_dataset["train"]
            vectorized_datasets["eval"] = split_dataset["test"]
            logger.info(f"Created eval split with {len(vectorized_datasets['eval'])} samples.")
        else:
            logger.error("No 'train' split found or too small to create 'eval' split from. Disabling evaluation.")
            training_args.do_eval = False

    # [FIX] Automatic capping for Kaggle to avoid OOM
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ and training_args.do_eval:
        if "eval" in vectorized_datasets:
            if data_args.max_eval_samples is None or data_args.max_eval_samples > 100:
                logger.warning("Kaggle environment detected. Capping evaluation samples to 50 for stability.")
                vectorized_datasets["eval"] = vectorized_datasets["eval"].select(range(min(len(vectorized_datasets["eval"]), 50)))

    # Initialize MusicgenTrainer
    trainer = MusicgenTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics if not getattr(training_args, "prediction_loss_only", False) else None,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets.get("eval") if training_args.do_eval else None,
        tokenizer=processor,
    )

    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:

        def decode_predictions(processor, predictions):
            audios = predictions.predictions
            return {"audio": np.array(audios.squeeze(1))}

        class WandbPredictionProgressCallback(WandbCallback):
            """
            Custom WandbCallback to log model predictions during training.
            """

            def __init__(
                self,
                trainer,
                processor,
                val_dataset,
                additional_generation,
                max_new_tokens,
                num_samples=8,
            ):
                """Initializes the WandbPredictionProgressCallback instance.

                Args:
                    trainer (Seq2SeqTrainer): The Hugging Face Seq2SeqTrainer instance.
                    processor (AutoProcessor): The processor associated
                    with the model.
                    val_dataset (Dataset): The validation dataset.
                    num_samples (int, optional): Number of samples to select from
                    the validation dataset for generating predictions.
                    Defaults to 8.
                """
                super().__init__()
                self.trainer = trainer
                self.processor = processor
                self.additional_generation = additional_generation
                self.sample_dataset = val_dataset.select(range(num_samples))
                self.max_new_tokens = max_new_tokens

            def on_train_begin(self, args, state, control, **kwargs):
                super().on_train_begin(args, state, control, **kwargs)
                set_seed(training_args.seed)
                predictions = self.trainer.predict(self.sample_dataset)
                # decode predictions and labels
                predictions = decode_predictions(self.processor, predictions)

                input_ids = self.sample_dataset["input_ids"]
                texts = self.processor.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                audios = [a for a in predictions["audio"]]

                additional_audio = self.trainer.model.generate(
                    **self.additional_generation.to(self.trainer.model.device),
                    max_new_tokens=self.max_new_tokens,
                )
                additional_text = self.processor.tokenizer.decode(
                    self.additional_generation["input_ids"].squeeze(),
                    skip_special_tokens=True,
                )
                texts.append(additional_text)
                audios.append(additional_audio.squeeze().cpu())

                # log the table to wandb
                self._wandb.log(
                    {
                        "sample_songs": [
                            self._wandb.Audio(
                                audio,
                                caption=text,
                                sample_rate=audio_encoder_feature_extractor.sampling_rate,
                            )
                            for (audio, text) in zip(audios, texts)
                        ]
                    }
                )

            def on_train_end(self, args, state, control, **kwargs):
                super().on_train_end(args, state, control, **kwargs)

                set_seed(training_args.seed)
                predictions = self.trainer.predict(self.sample_dataset)
                # decode predictions and labels
                predictions = decode_predictions(self.processor, predictions)

                input_ids = self.sample_dataset["input_ids"]
                texts = self.processor.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                audios = [a for a in predictions["audio"]]

                additional_audio = self.trainer.model.generate(
                    **self.additional_generation.to(self.trainer.model.device),
                    max_new_tokens=self.max_new_tokens,
                )
                additional_text = self.processor.tokenizer.decode(
                    self.additional_generation["input_ids"].squeeze(),
                    skip_special_tokens=True,
                )
                texts.append(additional_text)
                audios.append(additional_audio.squeeze().cpu())

                # log the table to wandb
                self._wandb.log(
                    {
                        "sample_songs": [
                            self._wandb.Audio(
                                audio,
                                caption=text,
                                sample_rate=audio_encoder_feature_extractor.sampling_rate,
                            )
                            for (audio, text) in zip(audios, texts)
                        ]
                    }
                )

        # Instantiate the WandbPredictionProgressCallback
        eval_dataset = vectorized_datasets.get("eval")
        if eval_dataset is not None and len(eval_dataset) > 0:
            num_samples = min(len(eval_dataset), data_args.num_samples_to_generate)
            progress_callback = WandbPredictionProgressCallback(
                trainer=trainer,
                processor=processor,
                val_dataset=eval_dataset,
                additional_generation=full_generation_sample_text,
                num_samples=num_samples,
                max_new_tokens=training_args.generation_max_length,
            )

            # Add the callback to the trainer
            trainer.add_callback(progress_callback)
        else:
            logger.warning("Evaluation dataset is missing or empty. Skipping WandbPredictionProgressCallback.")

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
            ignore_keys_for_eval=["past_key_values", "attentions"],
        )
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"])
        )

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(
            max_eval_samples, len(vectorized_datasets["eval"])
        )

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = (
        data_args.dataset_config_name
        if data_args.dataset_config_name is not None
        else "na"
    )
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-to-audio",
        "tags": ["text-to-audio", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    set_start_method("spawn")
    main()
