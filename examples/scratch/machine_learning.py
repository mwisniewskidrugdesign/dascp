import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # generyczny typ do reprezentowania punktów danych

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Podziel dane na zbiór treningowy i testowy."""
    data = data[:]                    # Zrób tzw. płytką kopię,
    random.shuffle(data)              # ponieważ funkcja shuffle modyfikuje listę.
    cut = int(len(data) * prob)       # Użyj parametru prob, aby znaleźć punkt podziału,
    return data[:cut], data[cut:]     # i rozdziel listę w tym punkcie.

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# Proporcje powinny się zgadzać
assert len(train) == 750
assert len(test) == 250

# A oryginalne dane powinny być zachowane (w odpowiednim porządku)
assert sorted(train + test) == data

Y = TypeVar('Y')  # generyczny typ do reprezentacji danych wyjściowych

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Tworzy indeksy i rozdziela je.
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

xs = [x for x in range(1000)]  # xs to wartości 1 … 1000
ys = [2 * x for x in xs]       # każda wartość y_i to podwojona wartość x_i
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# Sprawdzamy, czy proporcje się zgadzają
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# Sprawdzamy, czy odpowiednie punkty w danych są ze sobą sparowane 
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

assert accuracy(70, 4930, 13930, 981070) == 0.98114

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

assert recall(70, 4930, 13930, 981070) == 0.005

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)

