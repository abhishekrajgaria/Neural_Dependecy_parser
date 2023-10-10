import torch
import torchtext
import numpy as np
import pandas as pd
import dependecy_parser
from mp2_release.scripts import state


def getGloveEmbedding(gloVe_name, emb_dim):
    return torchtext.vocab.GloVe(name=gloVe_name, dim=emb_dim)


def getTokens(row):
    sentences = row["sentence"].split()
    poss = row["pos"].split()
    tokens = []
    for i, (word, pos) in enumerate(zip(sentences, poss)):
        token = state.Token(i, word, pos)
        tokens.append(token)
    return tokens


def loadTextFileToDF(fileName):
    df = pd.read_csv(
        fileName,
        sep="\s\|{3}\s",
        header=None,
        names=["sentence", "pos", "action"],
        engine="python",
    )
    df["tokens"] = df.apply(
        getTokens,
        axis=1,
    )
    df["actions"] = df.apply(lambda row: row["action"].split(), axis=1)
    return df[["tokens", "actions"]]


def loadDFWithoutActions(file_name):
    df = pd.read_csv(
        file_name,
        sep="\s\|{3}\s",
        header=None,
        names=["sentence", "pos"],
        engine="python",
    )
    df["tokens"] = df.apply(
        getTokens,
        axis=1,
    )

    return df["tokens"]


def context2idx(labelFileName):
    context2idx = {}
    with open(labelFileName) as f:
        for i, line in enumerate(f):
            context2idx[line.replace("\n", "")] = i
    return context2idx


def getData(file_name, emb_name, emb_size, with_label):
    parser = dependecy_parser.Parser(emb_name, emb_size)
    df = loadTextFileToDF(file_name)
    df["data"] = df.apply(
        lambda row: parser.directParser(row["tokens"], row["actions"], with_label),
        axis=1,
    )

    return df[["data"]]


def flattenData(dataframe):
    flatten_data = []
    for i, row in dataframe.iterrows():
        for input_label in row:
            flatten_data.extend(input_label)
    # print((flatten_data))
    # print("dataset size", len(flatten_data))
    return flatten_data


def getAllActionCount(filename):
    df = loadTextFileToDF(file_name)
    count = 0
    for i, row in df.iterrows():
        count += len(row["actions"])
    print("cnt", count)


def genDataset(filename, emb_name, emb_dim, with_label):
    dataframe = getData(filename, emb_name, emb_dim, with_label)
    flatten_data = flattenData(dataframe)
    word_rep = []
    pos_rep = []
    label_rep = []
    target = []
    for data in flatten_data:
        word_rep.append(data[0])
        pos_rep.append(data[1])
        label_rep.append(data[2])
        target.append(data[3])

    # pos_rep_tensor = torch.nn.Embedding(18, 50)(pos_rep_tensor).mean(dim=1)
    # print(pos_rep_tensor.shape)

    word_rep_tensor = torch.tensor(word_rep)
    pos_rep_tensor = torch.tensor(pos_rep)
    label_rep_tensor = torch.tensor(label_rep)
    target_tensor = torch.tensor(target)
    dataset = torch.utils.data.TensorDataset(
        word_rep_tensor, pos_rep_tensor, label_rep_tensor, target_tensor
    )
    # print(word_rep_tensor.shape)
    # print(pos_rep_tensor.shape)
    # print(label_tensor.shape)
    return dataset


if __name__ == "__main__":
    torch.manual_seed(42)
    pd.set_option("display.max_colwidth", None)  # Display the full content of each cell
    pd.set_option("display.width", None)  # Allow wide output
    # df = loadTextFileToDF("./mp2_release/data/train.txt")
    file_name = "./mp2_release/data/example.txt"
    # data = getData(file_name, "6B", 50)
    # flattenData(data)
    # getAllActionCount(file_name)
    genDataset(file_name, "6B", 50)
