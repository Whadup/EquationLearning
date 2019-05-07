# EquationLearning
We provide a dataset of equations represented as PNG-images and Latex code. This dataset is useful for learning to detect similarities in equations. Check it out [HERE!](https://whadup.github.io/EquationLearning/)

## Usage 

### Building the dataset(s)

You have to build the respective datasets before you can train or evaluate the equation-encoder.

To build the training dataset run:
```bash 
python formula_data.py train path/to/weak_data_train
```

To build the evaluation dataset (Gold-Label Evaluation Data) run:
```bash
python formula_data.py eval path/to/eval2
```

To build the evaluation dataset (Hold-Out Data) run:
```bash
python formula_data.py test path/to/weak_data_test
```

### Pretraining

In order to pretrain the equation-encoder run this:

```bash
python3 pretrain_experiment.py with dataset=task data_source=path/to/weak_data_train

```
```task``` should be either ```abstract``` or ```symbols``` depending on which pretraining task you want to run. 

### Training

In order to train the equation-encoder run this:

```bash
python equen_experiment.py
```

If you want to use weights from pretraining you should run something like:

```bash
python equen_experiment.py with pretrained_weights=path/to/weights
```
```path/to/weights``` should be something like ```equen_runs/x``` with x as the number of the respective training routine.

### Evaluation

In order to evaluate the trained weights from all epochs of a training routine run this:

```bash
python evaluation.py with run=path/to/run
```

```path/to/run``` should be something like ```equen_runs/x``` with x as the number of the respective training routine.

If you want to evaluate on Hold-Out data instead of the Gold-Label data you should run:

```bash
python evaluation.py with run=path/to/run dataset=test
```