from typing import List, Set


class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx  # Unique index of the token
        self.word = word  # Token string
        self.pos = pos  # Part of speech tag

    def __str__(self):
        return f"Token(idx={self.idx}, word='{self.word}', pos='{self.pos}')"

    def __eq__(self, other):
        return (self.idx, self.word, self.pos) == (other.idx, other.word, other.pos)


class DependencyEdge:
    def __init__(self, source: Token, target: Token, label: str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label = label  # dependency label
        pass


class ParseState:
    def __init__(
        self,
        stack: List[Token],
        parse_buffer: List[Token],
        dependencies: List[DependencyEdge],
    ):
        self.stack = stack  # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = parse_buffer  # A buffer of token indices
        self.dependencies = dependencies
        pass

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(state: ParseState) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    token = state.parse_buffer.pop(0)
    state.stack.append(token)


def left_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.

    token1 = state.stack.pop()
    token2 = state.stack.pop()
    state.add_dependency(token1, token2, label)
    state.stack.append(token1)


def right_arc(state: ParseState, label: str) -> None:
    # TODO: Implement this as an in-place operation that updates the parse state and does not return anything

    # The python documentation has some pointers on how lists can be used as stacks and queues. This may come useful:
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks
    # https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-queues

    # Also, you will need to use the state.add_dependency method defined above.

    token1 = state.stack.pop()
    token2 = state.stack.pop()
    state.add_dependency(token2, token1, label)
    state.stack.append(token2)


def is_final_state(state: ParseState, cwindow: int) -> bool:
    # TODO: Implemement this

    if len(state.stack) == 3 and len(state.parse_buffer) == 2:
        return True
    return False
