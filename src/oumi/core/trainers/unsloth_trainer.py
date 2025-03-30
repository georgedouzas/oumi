# Copyright 2025 - Oumi
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

from typing import Optional

import torch
from unsloth import FastLanguageModel

from oumi.core.configs.params.training_params import TrainingParams
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


class UnslothTrainer(BaseTrainer):
    """Trainer implementation using Unsloth's optimizations."""

    def __init__(
        self,
        model,
        tokenizer,
        training_args: TrainingParams,
        train_dataset=None,
        eval_dataset=None,
    ):
        """Initialize UnslothTrainer.

        Args:
            model: The model to train
            tokenizer: The tokenizer to use
            training_args: Training parameters
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _prepare_model_for_training(self):
        """Prepare the model for Unsloth training."""
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model=self.model,
                tokenizer=self.tokenizer,
                max_seq_length=self.model.config.max_position_embeddings,
                dtype=torch.bfloat16
                if self.training_args.mixed_precision_dtype == "bfloat16"
                else torch.float16,
                load_in_4bit=True,
            )
            self.model = model
            self.tokenizer = tokenizer
        except ImportError:
            logger.error(
                "Unsloth is not installed. Please install it with: pip install unsloth"
            )
            raise

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Trains the model using Unsloth's optimized training.

        Args:
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        pass

    def save_state(self) -> None:
        """Saves the trainer state."""
        # TODO: Implement state saving logic
        pass

    def save_model(self, config, final: bool = True) -> None:
        """Saves the model using Unsloth's specialized save methods.

        Args:
            config: The training configuration
            final: Whether this is the final save during training
        """
        pass
