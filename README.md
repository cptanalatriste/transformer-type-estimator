# transformer-type-estimator
A type estimator based on fine-tuning Google BERT

## Training

```bash
python transformer_analyser.py --train --train_csv "training_data.csv" --test_csv "testing_data.csv"
```

## Obtaining Probabilities

```bash
python transformer_analyser.py --pred --input_text "Of course I'll help\!"
```

## Training the Estimator

You can monitor training progress using TensorBoard. 
To start it, run the following command:

```bash
tensorboard --logdir=./logs --reload_multifile True
```

Then, open your browser at http://localhost:6006/ to see the Dashboard.