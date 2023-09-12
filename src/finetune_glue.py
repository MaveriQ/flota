import argparse
from morphpiece import MorphPieceBPE
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Optional
import datasets
import torch


class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        args: argparse.Namespace,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = args.model_name_or_path
        self.task_name = args.task_name
        self.max_seq_length = args.max_seq_length
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

        self.text_fields = self.task_text_field_map[self.task_name]
        self.num_labels = self.glue_task_num_labels[self.task_name]

        if 'morph' in self.model_name_or_path:
            self.tokenizer = MorphPieceBPE(model_max_length=self.max_seq_length)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        if self.model_name_or_path == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        # AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    
    @staticmethod
    def add_model_specific_args(parser):

        parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length")
        parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/TPU core/CPU for training.")
        parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size per GPU/TPU core/CPU for evaluation.")
        return parser

    
class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, 
                                                                        num_labels=num_labels, 
                                                                        pad_token_id=pad_token_id)

        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_labels", default=2, type=int, help="Number of labels to use in the last layer.")
        return parser
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_model_specific_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)

    parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
    parser.add_argument("--task_name", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="GLUE task name",
                        choices=['all']+GLUEDataModule.task_text_field_map.keys())

    tmp_args = "--model_name_or_path maveriq/morphgpt-base-200k --task_name mrpc".split()
    args = parser.parse_args()
    return args

def main(args):
    
    seed_everything(42)

    dm = GLUEDataModule(args)
    dm.setup("fit")
    
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=dm.train_batch_size,
        eval_batch_size=dm.eval_batch_size,
        pad_token_id=dm.tokenizer.pad_token_id,
    )

    checkpoint_callback = ModelCheckpoint(dirpath="/home/hj36wegi/scratch/data/morph/glue/checkpoints",
                                          save_top_k=3, 
                                          verbose=True, 
                                          filename=f"{args.model_name_or_path.split('/')[-1]}_{args.task_name}_{{epoch:02d}}_{{val_loss:.2f}}")
    
    logger = TensorBoardLogger(save_dir="/home/hj36wegi/scratch/data/morph/glue/lightning_logs", 
                               version=f"{args.model_name_or_path.split('/')[-1]}_{args.task_name}")

    trainer = Trainer(
                    max_epochs=3,
                    accelerator="auto",
                    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                    callbacks=[checkpoint_callback],
                    logger=logger,
                )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    args = parse_args()
    if args.task_name == 'all':
        for task in GLUEDataModule.task_text_field_map.keys():
            args.task_name = task
            main(args)
    else:
        main(args)