import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import util
import model
import mp2_release.scripts.state
import dependecy_parser
from mp2_release.scripts import evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, help="Directory where model checkpoints will be saved"
)
parser.add_argument("--emb_dim", type=int, help="word embedding dimension")
parser.add_argument("--emb_name", type=str, help="GloVe embedding dimension")
parser.add_argument(
    "--take_mean", type=str, default=False, help="True means take mean else concat"
)
parser.add_argument(
    "--with_label",
    type=str,
    default=False,
    help="False means do not include label else include label",
)
args = parser.parse_args()
output_dir = args.output_dir

best_perf_dict = {
    "metric": -1,
    "model_param": None,
    "optim_param": None,
    "epoch": 0,
    "learning_rate": 0,
}

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")


def loadModel(model_instance, model_path):
    checkpoint = torch.load(model_path)
    # print(checkpoint)
    model_instance.load_state_dict(checkpoint["model_param"])
    print(
        f"""Dev_LAS of loaded model: {checkpoint["dev_metric"]} at epoch {checkpoint["epoch"]} with learning rate {checkpoint["learning_rate"]}"""
    )
    return model_instance


def evaluateModel(parser, dataset, with_label):
    dataset["pred_actions"] = dataset.apply(
        lambda row: parser.actualParser(row["tokens"], with_label), axis=1
    )
    dataset["words"] = dataset.apply(
        lambda row: [token.word for token in row["tokens"]], axis=1
    )
    UAS, LAS = evaluate.compute_metrics(
        dataset["words"].tolist(),
        dataset["actions"].tolist(),
        dataset["pred_actions"].tolist(),
    )
    print("UAS", UAS, "LAS", LAS)
    return UAS, LAS


class Trainer:
    def __init__(
        self,
        model,
        parser,
        train_dataset,
        dev_dataset,
        batch_size,
        epochs,
        with_label=False,
    ) -> None:
        self.model = model
        self.parser = parser
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.with_label = with_label

    def evaluateModel(self, dataset):
        dataset["pred_actions"] = dataset.apply(
            lambda row: self.parser.actualParser(row["tokens"], self.with_label), axis=1
        )
        dataset["words"] = dataset.apply(
            lambda row: [token.word for token in row["tokens"]], axis=1
        )
        UAS, LAS = evaluate.compute_metrics(
            dataset["words"].tolist(),
            dataset["actions"].tolist(),
            dataset["pred_actions"].tolist(),
        )
        print("UAS", UAS, "LAS", LAS)
        return UAS, LAS

    def trainModel(self, learning_rate):
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

        train_dataLoader = torch.utils.data.DataLoader(
            self.train_dataset, self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            train_loss = []

            for (
                train_word_rep,
                train_pos_rep,
                train_label_rep,
                target,
            ) in train_dataLoader:
                self.model.train()
                optimizer.zero_grad()
                output = self.model(
                    train_word_rep.to(device),
                    train_pos_rep.to(device),
                    train_label_rep.to(device),
                )  # 1x75
                loss = lossFunction(output, target.to(device))
                train_loss.append(loss.cpu().item())

                loss.backward()  # computing the gradients
                optimizer.step()

            print(f"Average training batch loss: {np.mean(train_loss)}")

            _, LAS = self.evaluateModel(self.dev_dataset)
            if best_perf_dict["metric"] == -1 or LAS > best_perf_dict["metric"]:
                # print("hello")
                # best_perf_dict["model_param"]: self.model.state_dict()
                # best_perf_dict["optim_param"]: optimizer.state_dict()
                best_perf_dict["metric"] = LAS
                best_perf_dict["epoch"] = epoch
                best_perf_dict["learning_rate"] = learning_rate

                torch.save(
                    {
                        "model_param": self.model.state_dict(),
                        "optim_param": optimizer.state_dict(),
                        "dev_metric": LAS,
                        "epoch": epoch,
                        "learning_rate": best_perf_dict["learning_rate"],
                    },
                    f"{output_dir}/{emb_name}_{emb_dim}_{best_perf_dict['learning_rate']}_{best_perf_dict['epoch']}",
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    learning_rates = [0.01, 0.001, 0.0001]

    train_data_filename = "./mp2_release/data/train.txt"
    dev_data_filename = "./mp2_release/data/dev.txt"
    test_data_filename = "./mp2_release/data/test.txt"
    example_data_filename = "./mp2_release/data/example.txt"

    print(
        f"Arguments emb_dim {args.emb_dim}, emb_name {args.emb_name}, take_mean {args.take_mean}, with_label {args.with_label}, output_dir {args.output_dir}"
    )
    emb_name = args.emb_name
    emb_dim = args.emb_dim

    take_mean = True if args.take_mean == "yes" else False
    with_label = True if args.with_label == "yes" else False

    print(f"take_mean {take_mean}, with_label {with_label}")

    train_dataset = util.genDataset(train_data_filename, emb_name, emb_dim, with_label)
    dev_dataset = util.loadTextFileToDF(dev_data_filename)
    test_dataset = util.loadTextFileToDF(test_data_filename)

    # example_dataset = util.genDataset(
    #     example_data_filename, emb_name, emb_dim, with_label
    # )
    # example_dataset_2 = util.loadTextFileToDF(example_data_filename)

    model1 = model.MultiClassClassifier(
        word_emb_dim=emb_dim, take_mean=take_mean, with_label=with_label
    ).to(device)
    parser = dependecy_parser.Parser(
        emb_name=emb_name, emb_size=emb_dim, model=model1, device=device
    )

    for learning_rate in learning_rates:
        print("*****************")
        print("learning_rate", learning_rate)
        trainer = Trainer(
            model1,
            parser,
            train_dataset,
            dev_dataset,
            batch_size=64,
            epochs=20,
            with_label=with_label,
        )
        trainer.trainModel(learning_rate)
        print("*****************")
        print()

    best_model_path = f"{output_dir}/{emb_name}_{emb_dim}_{best_perf_dict['learning_rate']}_{best_perf_dict['epoch']}"
    print(best_model_path)
    # torch.save(
    #     {
    #         "model_param": best_perf_dict["model_param"],
    #         "optim_param": best_perf_dict["optim_param"],
    #         "dev_metric": best_perf_dict["metric"],
    #         "epoch": best_perf_dict["epoch"],
    #         "learning_rate": best_perf_dict["learning_rate"],
    #     },
    #     best_model_path,
    # )
    best_model = loadModel(model1, best_model_path)
    best_parser = dependecy_parser.Parser(
        emb_name=emb_name, emb_size=emb_dim, model=best_model, device=device
    )
    evaluateModel(best_parser, test_dataset, with_label)
    # trainer.evaluateModel(test_dataset)
