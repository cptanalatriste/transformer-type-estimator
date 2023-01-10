import datetime
import logging
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import tensorflow as tf
from datasets import Dataset, DatasetDict
from keras import Model
from keras.callbacks import Callback, TensorBoard, EarlyStopping
from transformers import BertTokenizerFast, AutoTokenizer, TFAutoModelForSequenceClassification, Pipeline, pipeline

GROUP_IDENTITY_CLASS: str = "LABEL_1"
TEXT_CONTENT_COLUMN = "text"
TEXT_LABEL_COLUMN = "will_help"
NON_ESSENTIAL_COLUMNS = ["title", "URL", "starting_time", "context", "logged_by"]


class TransformerTypeAnalyser(object):

    def __init__(self, epochs: int = 65, batch_size: int = 32, learning_rate: float = 3e-5,
                 output_directory: str = "./model"):
        self.model_checkpoint: str = "bert-base-uncased"
        self.hub_model_id: str = "cptanalatriste/request-for-help"
        self.pipeline_task: str = "text-classification"
        self.output_directory: str = output_directory
        self.log_directory: str = "logs/"

        self.model_input_names: List[str] = ['input_ids', 'token_type_ids', 'attention_mask']

        self.num_labels: int = 2
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
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
        if testing_data is not None:
            dataset_dict_input["test"] = Dataset.from_pandas(testing_data)
        dataset_dict: DatasetDict = DatasetDict(dataset_dict_input)

        logging.info("Encoding text ...")
        encoded_datasets: DatasetDict = dataset_dict.map(self.tokenize, batched=True)
        encoded_datasets = encoded_datasets.remove_columns(NON_ESSENTIAL_COLUMNS)

        training_dataset: tf.data.Dataset = self.convert_dataset_to_tensorflow(encoded_datasets["train"], shuffle=True)
        testing_dataset: Optional[tf.data.Dataset] = None

        logging.info(f"Encoding finished. Starting training")
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

    def obtain_probabilities(self, text_data: Union[str, List[str]], local=False) -> List[float]:
        classification_pipeline: Pipeline
        if local:
            logging.info(f"Loading model from {self.output_directory}")
            classification_pipeline = pipeline(task=self.pipeline_task,
                                               model=self.output_directory,
                                               tokenizer=self.output_directory)
        else:
            classification_pipeline = pipeline(task=self.pipeline_task, model=self.hub_model_id)

        predictions: List[List[Dict]] = classification_pipeline(text_data, return_all_scores=True)

        group_identity_probabilities: List[float] = [extract_group_probability(prediction_items) for prediction_items in
                                                     predictions]

        return group_identity_probabilities


def get_log_directory(base_directory: str = "logs/") -> str:
    return base_directory + datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")


def extract_group_probability(prediction_items: List[Dict[str, Any]]) -> Optional[float]:
    for prediction_item in prediction_items:
        if prediction_item["label"] == GROUP_IDENTITY_CLASS:
            return prediction_item["score"]

    return None
