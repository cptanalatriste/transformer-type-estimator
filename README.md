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