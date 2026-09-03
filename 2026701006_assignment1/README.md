# Advanced NLP Assignment 1

## Setup

Download and extract the provided ZIP folder.

Create the environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` contains the frozen dependencies used for the experiments.

After setting up the environment, the experiments can be run using the functions and configurations provided in `train.py` and `dataset.py`.

`dataset.py` handles dataset preparation and DataLoader construction for the respective BPE and BLT configurations. The Transformer architecture is implemented using the attention, normalization, and positional components in `models`. `transformer.py` acts as a bridge between these model components and `train.py` for convenient model construction and use. The BLT-specific implementation is contained in `blt.py`. Training and experiment execution are handled by `train.py`, with evaluation metrics provided in `utils.py`.

## Hugging Face

* **Models / Checkpoints:** <https://huggingface.co/vibhav20/anlp-a1-transformer/tree/main>
