import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import util
import main
import model
import mp2_release.scripts.state
import dependecy_parser

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")


def writeActionToFile(row):
    # print(row)
    with open("report_sentence_result.txt", "a", encoding="utf-8") as f:
        f.write(" ".join(row))
        f.write("\n")


def getPrediction(parser: dependecy_parser.Parser, dataset):
    predicted_dataset = dataset.apply(lambda row: parser.actualParser(row))
    # print(predicted_dataset.head)
    predicted_dataset.apply(writeActionToFile)


if __name__ == "__main__":
    torch.manual_seed(42)
    # pd.set_option("display.max_colwidth", None)  # Display the full content of each cell
    pd.set_option("display.width", None)  # Allow wide output

    learning_rate = 0.0001

    hidden_filename = "./mp2_release/data/report_sentence.txt"

    model_dir = "/scratch/general/vast/u1471428/cs6957/assignment2/models/concat"
    emb_name = "840B"
    emb_dim = 300

    take_mean = False
    with_label = False

    hidden_dataset = util.loadDFWithoutActions(hidden_filename)

    # print(hidden_dataset.head)

    model1 = model.MultiClassClassifier(
        word_emb_dim=emb_dim, take_mean=take_mean, with_label=with_label
    ).to(device)

    best_model_path = f"{model_dir}/{emb_name}_{emb_dim}_{learning_rate}_12"

    best_model = main.loadModel(model1, best_model_path)
    best_parser = dependecy_parser.Parser(
        emb_name=emb_name, emb_size=emb_dim, model=best_model, device=device
    )

    getPrediction(best_parser, hidden_dataset)
    # trainer.evaluateModel(test_dataset)
