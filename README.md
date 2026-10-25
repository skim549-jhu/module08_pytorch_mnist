# module08_pytorch_mnist

MNIST Classification with PyTorch (Module 7, Lecture B, Section 3) rewritten as an Object Oriented Programming (OOP) project with `argparse`, developed in GitHub Codespaces (web-based Visual Studio Code).

## Files

| File | Description |
|---|---|
| `utilities.py` | `get_args()` (argparse), `ConvNet` (model), `MNISTClassifier` (data, loss, optimizer, & training loop), & `plot()` |
| `run.py` | Trains the model from the terminal & saves the plot |
| `requirements.txt` | Required Python libraries |
| `.gitignore` | Excludes `__pycache__/` & the downloaded `data/` folder |

## How to Run

```sh
pip install -r requirements.txt
python run.py
```

| Short flag | Long flag | Description | Default |
|---|---|---|---|
| `-e` | `--num_epochs` | Number of epochs | 2 |
| `-b` | `--batch_size` | Batch size | 1000 |
| `-l` | `--learning_rate` | Learning rate of the Adam optimizer | 0.001 |

Examples:

```sh
python run.py -e 3 -b 500 -l 0.0005
python run.py --num_epochs 1 --batch_size 2000 --learning_rate 0.002
python run.py -h
```

## Output

* Progress bars with the running training loss & the training/testing accuracy of each epoch in the terminal.
* `figure.png`: the average training loss of each epoch (left) & the training/testing accuracy of each epoch (right).

> Note: As of Aug 2024, GitHub Codespaces do not offer free GPUs, so the model trains on the CPU there. The same code uses a GPU automatically when one is available.
