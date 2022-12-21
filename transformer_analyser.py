import os

from torch import Tensor

SEED: int = 42
import numpy as np

np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
# physical_devices = tf.config.list_physical_devices("GPU")
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import logging
from argparse import ArgumentParser
from typing import Optional, List, Dict

import pandas as pd
from datasets import Dataset, DatasetDict
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from transformers import AutoTokenizer, BertTokenizerFast, TFAutoModelForSequenceClassification, \
    DataCollatorWithPadding, pipeline, Pipeline

GROUP_IDENTITY_CLASS: str = "LABEL_1"
TEXT_CONTENT_COLUMN = "text"
TEXT_LABEL_COLUMN = "label"
NON_ESSENTIAL_COLUMNS = ["title", "URL", "starting_time", "context", "logged_by", "Observation"]


class TransformerTypeAnalyser(object):

    def __init__(self):
        self.model_checkpoint: str = "bert-base-uncased"
        self.hub_model_id: str = "cptanalatriste/request-for-help"
        self.pipeline_task: str = "text-classification"
        self.output_directory: str = "./model"
        self.model_input_names: List[str] = ['input_ids', 'token_type_ids', 'attention_mask']

        self.num_labels: int = 2
        self.batch_size: int = 32
        self.learning_rate: float = 3e-5
        self.epochs = 65
        self.early_stopping_patience: int = int(self.epochs / 3)

        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[Model] = None

    def tokenize(self, data_batch):
        return self.tokenizer(data_batch[TEXT_CONTENT_COLUMN], truncation=True, padding="max_length")

    @staticmethod
    def convert_csv_to_dataset(csv_file: str) -> Dataset:
        dataframe: pd.DataFrame = pd.read_csv(csv_file)
        dataset: Dataset = Dataset.from_pandas(dataframe)

        return dataset

    def convert_dataset_to_tensorflow(self, encoded_datasets: Dataset, shuffle: bool) -> tf.data.Dataset:
        tensorflow_dataset: Dataset = encoded_datasets.remove_columns([TEXT_CONTENT_COLUMN]).with_format("tensorflow")

        features: Dict[str, Tensor] = {input_name: tensorflow_dataset[input_name]
                                       for input_name in self.model_input_names}
        result_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
            (features, tensorflow_dataset[TEXT_LABEL_COLUMN]))
        if shuffle:
            result_dataset = result_dataset.shuffle(len(tensorflow_dataset)).batch(self.batch_size)
        else:
            result_dataset = result_dataset.batch(self.batch_size)

        return result_dataset

    def train(self, training_data_file: str, testing_data_file: str, local=False):
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

        logging.info("Encoding text ...")
        encoded_datasets: DatasetDict = dataset_dict.map(self.tokenize, batched=True)
        encoded_datasets = encoded_datasets.remove_columns(NON_ESSENTIAL_COLUMNS)

        training_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["train"], shuffle=True)
        testing_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["test"], shuffle=False)

        logging.info(f"Encoding finished!")

        early_stopping_callback: EarlyStopping = EarlyStopping(monitor='val_loss',
                                                               patience=self.early_stopping_patience,
                                                               verbose=True,
                                                               restore_best_weights=True)

        if local:
            self.model.fit(training_dataset, validation_data=testing_dataset, epochs=self.epochs,
                           callbacks=[early_stopping_callback])
            self.model.save_pretrained(self.output_directory)
        else:
            # push_to_hub_callback: PushToHubCallback = PushToHubCallback(output_dir=self.output_directory,
            #                                                             tokenizer=self.tokenizer,
            #                                                             hub_model_id=self.hub_model_id)
            #
            # self.model.fit(training_dataset, validation_data=testing_dataset, epochs=self.epochs,
            #                callbacks=[early_stopping_callback, push_to_hub_callback])
            #
            # logging.info(f"Training finished! Model is available at the hub with id {self.hub_model_id}")
            pass

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


def start_training(training_csv: str, testing_csv_file: str, local: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"training_csv {training_csv}")
    logging.info(f"testing_csv_file {testing_csv_file}")

    type_analyser: TransformerTypeAnalyser = TransformerTypeAnalyser()
    type_analyser.train(training_csv, testing_csv_file, local)


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
        start_training(arguments.train_csv, arguments.test_csv, local=False)
    elif arguments.trainlocal:
        start_training(arguments.train_csv, arguments.test_csv, local=True)
    elif arguments.pred:
        obtain_probabilities(arguments.input_text, local=False)
    elif arguments.predlocal:
        obtain_probabilities(arguments.input_text, local=True)


if __name__ == "__main__":
    main()
