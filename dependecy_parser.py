import util
import torch
import torchtext
import constants
import numpy as np
import pandas as pd
from collections import deque
from mp2_release.scripts import state

import model

pad_token = state.Token(0, "NULL", "NULL")


def mapValuesToKey(dict):
    key_list = list(dict.keys())
    val_list = list(dict.values())

    new_dict = {}
    for value, key in zip(val_list, key_list):
        new_dict[value] = key

    return new_dict


def getLabelWithoutAction(keys):
    idx = 0
    new_dict = {}
    for key in keys:
        if key != "SHIFT" and key[9:] not in new_dict:
            new_dict[key[9:]] = idx
            idx += 1
    new_dict["NULL"] = idx
    # print(new_dict)
    return new_dict


class Parser:
    def __init__(
        self,
        emb_name,
        emb_size,
        model: model.MultiClassClassifier = None,
        device=torch.device("cpu"),
    ):
        self.gloVe_emb = util.getGloveEmbedding(emb_name, emb_size)
        self.device = device
        self.model = model
        self.emb_dim = emb_size
        self.window_size = constants.C_WINDOW
        self.pos2idx = util.context2idx("./mp2_release/data/pos_set.txt")
        self.label2idx = util.context2idx("./mp2_release/data/tagset.txt")
        self.idx2label = mapValuesToKey(self.label2idx)
        self.labelWithoutAction2idx = getLabelWithoutAction(self.label2idx.keys())

    def getWordRepresentation(self, parse_state: state.ParseState):
        stack_tokens = parse_state.stack[-self.window_size :]
        buffer_tokens = parse_state.parse_buffer[: self.window_size]

        tokens = stack_tokens + buffer_tokens

        words = [token.word for token in tokens]
        word_representations = self.gloVe_emb.get_vecs_by_tokens(words).tolist()
        # print(word_representations.shape)
        return word_representations

    def getPosRepresentation(self, parse_state):
        stack_tokens = parse_state.stack[-self.window_size :]
        # (print(tokens) for tokens in stack_tokens)
        buffer_tokens = parse_state.parse_buffer[: self.window_size]

        tokens = stack_tokens + buffer_tokens

        poss = [token.pos for token in tokens]
        pos_representations = [self.pos2idx[pos] for pos in poss]
        return pos_representations

    def getLeftMost(self, token: state.Token, dependencies):
        left_most = pad_token
        left_most_label = "NULL"
        left_most_index = -1
        for dependency_edge in dependencies:
            if dependency_edge.source == token:
                target = dependency_edge.target
                if target.idx < token.idx:
                    if left_most_index == -1 or left_most_index > target.idx:
                        left_most = target
                        left_most_index = target.idx
                        left_most_label = dependency_edge.label

        return left_most.word, left_most_label

    def getRightMost(self, token: state.Token, dependencies):
        right_most = pad_token
        right_most_label = "NULL"
        right_most_index = -1
        for dependency_edge in dependencies:
            if dependency_edge.source == token:
                target = dependency_edge.target
                if target.idx > token.idx:
                    # print(target, token)
                    if right_most_index == -1 or right_most_index < target.idx:
                        right_most = target
                        right_most_index = target.idx
                        right_most_label = dependency_edge.label

        return right_most.word, right_most_label

    def getDepLabelRepresentation(self, parse_state: state.ParseState):
        stack_tokens = parse_state.stack[-self.window_size :]
        words = []
        labels = []
        for token in stack_tokens:
            # print(token)
            leftmost_child, leftmost_child_label = self.getLeftMost(
                token, parse_state.dependencies
            )
            rightmost_child, rightmost_child_label = self.getRightMost(
                token, parse_state.dependencies
            )
            words += [leftmost_child, rightmost_child]
            labels += [leftmost_child_label, rightmost_child_label]

        word_representations = self.gloVe_emb.get_vecs_by_tokens(words).tolist()
        labels = [self.labelWithoutAction2idx[label] for label in labels]
        return word_representations, labels

    def isValidAction(self, parse_state: state.ParseState, action):
        if action == "SHIFT" and len(parse_state.parse_buffer) > 2:
            return True
        elif action != "SHIFT" and len(parse_state.stack) > 3:
            return True

        return False

    def getActionPrediction1(self, parse_state):
        # print("Action Prediction 1")
        word_representations = torch.tensor(self.getWordRepresentation(parse_state))
        # word_representations.view(-1, 2 * self.window_size, self.emb_dim)
        word_representations = torch.unsqueeze(word_representations, dim=0)
        # print(word_representations.shape)
        pos_representations = torch.tensor(self.getPosRepresentation(parse_state))
        # pos_representations.view(-1, 2 * self.window_size)
        pos_representations = torch.unsqueeze(pos_representations, dim=0)
        # print(pos_representations.shape)
        output = self.model.forward(
            word_representations.to(self.device), pos_representations.to(self.device)
        )
        # print(output.shape)
        pred = torch.topk(output.flatten(), 75).indices.tolist()
        # print(pred)
        valid_top = 0
        while not self.isValidAction(parse_state, self.idx2label[pred[valid_top]]):
            valid_top += 1
        # print("action", self.idx2label[pred[valid_top]])
        return self.idx2label[pred[valid_top]]

    def getActionPrediction2(self, parse_state):
        # print("Action Prediction 2")
        (
            additional_word_representation,
            label_representations,
        ) = self.getDepLabelRepresentation(parse_state)
        word_representations = self.getWordRepresentation(parse_state)
        pos_representations = self.getPosRepresentation(parse_state)

        word_representations += additional_word_representation
        word_representations = torch.tensor(word_representations)
        # word_representations.view(-1, 2 * self.window_size, self.emb_dim)
        word_representations = torch.unsqueeze(word_representations, dim=0)
        # print(word_representations.shape)
        pos_representations = torch.tensor(pos_representations)
        # pos_representations.view(-1, 2 * self.window_size)
        pos_representations = torch.unsqueeze(pos_representations, dim=0)
        # print(pos_representations.shape)

        label_representations = torch.tensor(label_representations)
        # pos_representations.view(-1, 2 * self.window_size)
        label_representations = torch.unsqueeze(label_representations, dim=0)
        # print(pos_representations.shape)

        output = self.model.forward(
            word_representations.to(self.device),
            pos_representations.to(self.device),
            label_representations.to(self.device),
        )
        # print(output.shape)
        pred = torch.topk(output.flatten(), 75).indices.tolist()
        # print(pred)
        valid_top = 0
        while not self.isValidAction(parse_state, self.idx2label[pred[valid_top]]):
            valid_top += 1
        # print("action", self.idx2label[pred[valid_top]])
        return self.idx2label[pred[valid_top]]

    # return (input, label) pairs for a sentence
    def directParser(self, sentence, actions, with_label=False):
        step = 0
        # pad_token = state.Token(0, "PAD", "NULL")
        stack = [pad_token for i in range(self.window_size)]
        parse_buffer = [token for token in sentence]
        parse_buffer.extend([pad_token for i in range(self.window_size)])

        input_label = []

        parse_state = state.ParseState(
            stack=stack, parse_buffer=parse_buffer, dependencies=[]
        )
        while not state.is_final_state(parse_state, self.window_size):
            action = actions[step]
            step += 1
            word_representations = self.getWordRepresentation(parse_state)
            pos_representations = self.getPosRepresentation(parse_state)

            additional_word_representation = []
            label_representation = []
            # print("with_label", with_label)
            if with_label == True:
                (
                    additional_word_representation,
                    label_representation,
                ) = self.getDepLabelRepresentation(parse_state)

            target = self.label2idx[action]
            # print(len(word_representations), len(additional_word_representation))
            input_label.append(
                [
                    word_representations + (additional_word_representation),
                    pos_representations,
                    label_representation,
                    target,
                ]
            )
            # print(len(input_label[0][0]))
            # label.append(target)

            if action == "SHIFT":
                state.shift(parse_state)
            elif action[:8] == "REDUCE_L":
                state.left_arc(parse_state, action[9:])
            else:
                state.right_arc(parse_state, action[9:])

        # print(len(input_label))
        return input_label

    # return set of actions for a sentence
    def actualParser(self, sentence, with_label=False):
        step = 0
        actions = []
        # pad_token = state.Token(0, "PAD", "NULL")
        stack = [pad_token for i in range(self.window_size)]
        parse_buffer = [token for token in sentence]
        parse_buffer.extend([pad_token for i in range(self.window_size)])
        parse_state = state.ParseState(
            stack=stack, parse_buffer=parse_buffer, dependencies=[]
        )
        while not state.is_final_state(parse_state, self.window_size):
            # print("step", step)
            # step += 1
            action = (
                self.getActionPrediction2(parse_state)
                if with_label
                else self.getActionPrediction1(parse_state)
            )
            if action == "SHIFT":
                state.shift(parse_state)
            elif action[:8] == "REDUCE_L":
                state.left_arc(parse_state, action[9:])
            else:
                state.right_arc(parse_state, action[9:])

            actions.append(action)

        # print("sentence ", sentence)
        # print("actions", actions)
        return actions


# def genData(file_name, emb_name, emb_size):
#     parser = Parser(emb_name, emb_size)
#     df = util.loadTextFileToDF(file_name)
#     df["data"] = df.apply(
#         lambda row: parser.directParser(row["tokens"], row["actions"], True),
#         axis=1,
#     )

#     return df[["data"]]


if __name__ == "__main__":
    torch.manual_seed(42)
    pd.set_option("display.max_colwidth", None)  # Display the full content of each cell
    pd.set_option("display.width", None)  # Allow wide output
    # file_name = "./mp2_release/data/example.txt"
    # data = genData(file_name, "6B", 50)
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    # print(data)
