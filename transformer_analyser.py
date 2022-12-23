import datetime

SEED: int = 42
import numpy as np

np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)

import logging
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any

import pandas as pd
from datasets import Dataset, DatasetDict
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras import Model
from transformers import AutoTokenizer, BertTokenizerFast, TFAutoModelForSequenceClassification, \
    pipeline, Pipeline

GROUP_IDENTITY_CLASS: str = "LABEL_1"
TEXT_CONTENT_COLUMN = "text"
TEXT_LABEL_COLUMN = "will_help"
NON_ESSENTIAL_COLUMNS = ["title", "URL", "starting_time", "context", "logged_by"]


class TransformerTypeAnalyser(object):

    def __init__(self, epochs=65):
        self.model_checkpoint: str = "bert-base-uncased"
        self.hub_model_id: str = "cptanalatriste/request-for-help"
        self.pipeline_task: str = "text-classification"
        self.output_directory: str = "./model"
        self.log_directory: str = "logs/"

        self.model_input_names: List[str] = ['input_ids', 'token_type_ids', 'attention_mask']

        self.num_labels: int = 2
        self.batch_size: int = 32
        self.learning_rate: float = 3e-5
        self.epochs = epochs
        self.early_stopping_patience: int = int(self.epochs / 3)

        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[Model] = None

    def tokenize(self, data_batch):
        return self.tokenizer(data_batch[TEXT_CONTENT_COLUMN], truncation=True, padding="max_length")

    def convert_dataset_to_tensorflow(self, encoded_datasets: Dataset, shuffle: bool) -> tf.data.Dataset:
        tensorflow_dataset: Dataset = encoded_datasets.remove_columns([TEXT_CONTENT_COLUMN]).with_format("tensorflow")

        features: Dict[str, Any] = {input_name: tensorflow_dataset[input_name]
                                    for input_name in self.model_input_names}
        result_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
            (features, tensorflow_dataset[TEXT_LABEL_COLUMN]))
        if shuffle:
            result_dataset = result_dataset.shuffle(len(tensorflow_dataset)).batch(self.batch_size)
        else:
            result_dataset = result_dataset.batch(self.batch_size)

        return result_dataset

    def train_from_files(self, training_data_file: str, testing_data_file: Optional[str] = None):
        self.train(pd.read_csv(training_data_file), pd.read_csv(testing_data_file))

    def train(self, training_data: pd.DataFrame, testing_data: Optional[pd.DataFrame] = None):
        logging.info("Starting training...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_checkpoint,
                                                                          num_labels=self.num_labels)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.metrics.SparseCategoricalAccuracy())
        logging.info("Model compiled...")

        dataset_dict_input: Dict[str, Dataset] = {
            "train": Dataset.from_pandas(training_data)
        }
        if testing_data:
            dataset_dict_input["test"] = Dataset.from_pandas(testing_data)
        dataset_dict: DatasetDict = DatasetDict(dataset_dict_input)

        logging.info("Encoding text ...")
        encoded_datasets: DatasetDict = dataset_dict.map(self.tokenize, batched=True)
        encoded_datasets = encoded_datasets.remove_columns(NON_ESSENTIAL_COLUMNS)

        training_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["train"], shuffle=True)
        testing_dataset: Optional[tf.data.Dataset] = None

        logging.info(f"Encoding finished!")
        training_callbacks: List[Callback] = [TensorBoard(log_dir=get_log_directory(self.log_directory))]

        if "test" in encoded_datasets:
            testing_dataset = self.convert_dataset_to_tensorflow(encoded_datasets["test"], shuffle=False)
            training_callbacks.append(EarlyStopping(monitor='val_loss',
                                                    patience=self.early_stopping_patience,
                                                    verbose=True,
                                                    restore_best_weights=True))

        self.model.fit(training_dataset,
                       validation_data=testing_dataset,
                       epochs=self.epochs,
                       callbacks=training_callbacks)
        self.model.save_pretrained(self.output_directory)

        self.tokenizer.save_pretrained(self.output_directory)
        logging.info(f"Model and Tokenizer saved at {self.output_directory}")

    def obtain_probabilities(self, text_as_string: str, local=False) -> float:
        classification_pipeline: Pipeline
        if local:
            logging.info(f"Loading model from {self.output_directory}")
            classification_pipeline = pipeline(task=self.pipeline_task,
                                               model=self.output_directory,
                                               tokenizer=self.output_directory)
        else:
            classification_pipeline = pipeline(task=self.pipeline_task, model=self.hub_model_id)

        predictions: List[List[Dict]] = classification_pipeline(text_as_string, return_all_scores=True)

        group_identity_probability: float = [entry["score"] for entry in predictions[0]
                                             if entry["label"] == GROUP_IDENTITY_CLASS][0]

        return group_identity_probability


def get_log_directory(base_directory: str = "logs/") -> str:
    return base_directory + datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")


def start_training(training_csv: str, testing_csv_file: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"training_csv {training_csv}")
    logging.info(f"testing_csv_file {testing_csv_file}")

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train_from_files(training_csv, testing_csv_file)


def obtain_probabilities(input_text: str, local: bool):
    logging.basicConfig(level=logging.DEBUG)
    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    prediction: float = type_analyser.obtain_probabilities(input_text, local)
    print(prediction)


def main():
    parser: ArgumentParser = ArgumentParser(
        description="A transformer-based intent recognition for answers to help requests.")
    parser.add_argument("--train_csv", type=str, help="CSV file with training data.")
    parser.add_argument("--test_csv", type=str, help="CSV file with testing data.")
    parser.add_argument("--input_text", type=str, help="Input text, for probability calculation.")
    parser.add_argument("--train", action="store_true", help="Start fine-tuning the pre-trained transformer model and "
                                                             "sending it to the hub.")
    parser.add_argument("--trainlocal", action="store_true", help="Start fine-tuning the pre-trained transformer model"
                                                                  " and saving it locally.")
    parser.add_argument("--pred", action="store_true", help="Calculate the probability for helping given a text"
                                                            " using a remote model.")
    parser.add_argument("--predlocal", action="store_true", help="Calculate the probability for helping given a text"
                                                                 " using a local model.")
    arguments = parser.parse_args()
    if arguments.train:
        start_training(arguments.train_csv, arguments.test_csv)
    elif arguments.trainlocal:
        start_training(arguments.train_csv, arguments.test_csv)
    elif arguments.pred:
        obtain_probabilities(arguments.input_text, local=False)
    elif arguments.predlocal:
        obtain_probabilities(arguments.input_text, local=True)


if __name__ == "__main__":
    main()
