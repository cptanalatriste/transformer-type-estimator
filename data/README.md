## Data Files

The file `Emergency_Expressions.xlsx` contains
all the utterances collected from online
emergency videos.
Two researchers assigned a label to each
utterance, indicating if it corresponds to
shared identity adoption (`Group`) or
not (`No_Group`).
The labels per researcher are listed in
the columns `Marker_1` and `Marker_2`.

## Training, Validation and Testing
The `Emergency_Expressions.xlsx` file was initially
partitioned in two files: `training_data.csv` to be used
during training-calibration of the type estimator and
`testing_data.csv` for evaluation purposes.
The code for this initial partition is contained in
the function `load_and_split()` within the `nlp_runner.py` 
file.

The training and calibration process requires to
further partition the `training_data.csv` into 
a dataset for training and another one for validation
and calibration. 
Please check the `training_type_estimator.ipynb` 
notebook for details on this process.