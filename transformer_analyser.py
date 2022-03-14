import logging
from argparse import ArgumentParser
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset, DatasetDict
from keras import Model
from transformers import AutoTokenizer, BertTokenizerFast, TFAutoModelForSequenceClassification, \
    DataCollatorWithPadding, PushToHubCallback, pipeline, Pipeline

SEED: int = 0

GROUP_IDENTITY_CLASS: str = "LABEL_1"


class TransformerTypeAnalyser(object):

    def __init__(self):
        self.model_checkpoint: str = "bert-base-uncased"
        self.text_column: str = "text"
        self.label_column: str = "will_help"
        self.irrelevant_columns = ["tweet_id", "wont_help"]
        self.hub_model_id: str = "cptanalatriste/request-for-help"
        self.pipeline_task: str = "text-classification"

        self.num_labels: int = 2
        self.batch_size: int = 32
        self.learning_rate: float = 3e-5
        self.epochs = 20

        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[Model] = None

    def tokenize(self, data_batch):
        return self.tokenizer(data_batch[self.text_column], truncation=True, max_length=128)

    def convert_csv_to_dataset(self, csv_file: str) -> Dataset:
        dataframe: pd.DataFrame = pd.read_csv(csv_file)
        dataset: Dataset = Dataset.from_pandas(dataframe)

        return dataset

    def convert_dataset_to_tensorflow(self, dataset: Dataset, shuffle: bool) -> tf.data.Dataset:
        data_collator: DataCollatorWithPadding = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        tensorflow_dataset: tf.data.Dataset = dataset.to_tf_dataset(columns=self.tokenizer.model_input_names,
                                                                    label_cols=[self.label_column],
                                                                    shuffle=shuffle,
                                                                    batch_size=self.batch_size,
                                                                    collate_fn=data_collator)

        return tensorflow_dataset

    def train(self, training_data_file: str, testing_data_file: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_checkpoint,
                                                                          num_labels=self.num_labels)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.metrics.SparseCategoricalAccuracy())

        dataset_dict: DatasetDict = DatasetDict({
            "train": self.convert_csv_to_dataset(training_data_file),
            "test": self.convert_csv_to_dataset(testing_data_file)
        })

        logging.info("Encoding content for tweets")
        encoded_datasets: DatasetDict = dataset_dict.map(self.tokenize, batched=True)
        encoded_datasets = encoded_datasets.remove_columns(self.irrelevant_columns)

        training_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["train"], shuffle=True)
        testing_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["test"], shuffle=False)

        logging.info(f"Encoding finished!")

        push_to_hub_callback: PushToHubCallback = PushToHubCallback(output_dir="./model",
                                                                    tokenizer=self.tokenizer,
                                                                    hub_model_id=self.hub_model_id)

        self.model.fit(training_dataset, validation_data=testing_dataset, epochs=self.epochs,
                       callbacks=push_to_hub_callback)

        logging.info(f"Training finished! Model is available at the hub with id {self.hub_model_id}")

    def obtain_probabilities(self, text_as_string: str) -> float:
        classification_pipeline: Pipeline = pipeline(self.pipeline_task, self.hub_model_id)
        predictions: List[List[Dict]] = classification_pipeline(text_as_string, return_all_scores=True)

        group_identity_probability: float = [entry["score"] for entry in predictions[0]
                                             if entry["label"] == GROUP_IDENTITY_CLASS][0]

        return group_identity_probability


def start_training(training_csv: str, testing_csv_file: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"training_csv {training_csv}")
    logging.info(f"testing_csv_file {testing_csv_file}")

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train(training_csv, testing_csv_file)


def obtain_probabilities(input_text: str):
    logging.basicConfig(level=logging.DEBUG)
    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    prediction: float = type_analyser.obtain_probabilities(input_text)
    print(prediction)


def main():
    np.random.seed(SEED)

    parser: ArgumentParser = ArgumentParser(
        description="A transformer-based intent recognition for answers to help requests.")
    parser.add_argument("--train_csv", type=str, help="CSV file with training data.")
    parser.add_argument("--test_csv", type=str, help="CSV file with testing data.")
    parser.add_argument("--input_text", type=str, help="Input text, for probability calculation.")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning the pre-trained transformer model.")
    parser.add_argument("--pred", action="store_true", help="Calculate the probability for helping given a text.")

    arguments = parser.parse_args()
    if arguments.train:
        start_training(arguments.train_csv, arguments.test_csv)

    if arguments.pred:
        obtain_probabilities(arguments.input_text)


if __name__ == "__main__":
    main()
