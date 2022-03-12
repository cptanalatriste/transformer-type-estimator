import logging
from typing import Optional

import pandas as pd
from keras import Model
from transformers import AutoTokenizer, BertTokenizerFast, TFAutoModelForSequenceClassification, \
    DataCollatorWithPadding, PushToHubCallback
from datasets import Dataset, DatasetDict
import tensorflow as tf


class TransformerTypeAnalyser(object):

    def __init__(self, model_file: Optional[str] = None):
        self.model_checkpoint: str = "bert-base-uncased"
        self.text_column: str = "text"
        self.label_column: str = "will_help"
        self.irrelevant_columns = ["tweet_id", "wont_help"]
        self.hub_model_id: str = "cptanalatriste/request-for-help"

        self.num_labels: int = 2
        self.batch_size: int = 32
        self.learning_rate: float = 3e-5
        self.epochs = 20

        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model: Model = TFAutoModelForSequenceClassification.from_pretrained(self.model_checkpoint,
                                                                                 num_labels=self.num_labels)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.metrics.SparseCategoricalAccuracy())

    def tokenize(self, data_batch):
        return self.tokenizer(data_batch[self.text_column], truncation=True, max_length=128)

    def convert_csv_to_dataset(self, csv_file: str) -> Dataset:
        dataframe: pd.DataFrame = pd.read_csv(csv_file)
        dataframe[self.label_column] = dataframe[self.label_column].astype(int)
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

    def obtain_probabilities(self, text_as_string: str) -> float:
        pass
