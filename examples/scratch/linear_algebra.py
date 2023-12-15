from typing import List

Vector = List[float]

wzrost_waga_wiek = [170,  # centymetrów,
                    70,   # kilogramów,
                    40 ]  # lat


oceny  = [95,   # egzamin1
          80,   # egzamin2
          75,   # egzamin3
          62 ]  # egzamin4

def add(v: Vector, w: Vector) -> Vector:
    """dodawanie wektorów"""
    assert len(v) == len(w), "wektory muszą mieć tę samą długość"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """odejmowanie wektorów"""
    assert len(v) == len(w), "wektory muszą mieć tę samą długość "

    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sumuje listę wektorów"""
    # Sprawdzenie, czy lista wektorów nie jest pusta
    assert vectors, "brak wektorów!"

    # Sprawdzenie, czy wszystkie wektory mają taką samą długość
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "różne długości!"

    # i-ty element wektora wynikowego jest sumą elementów [i] każdego wektora
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Mnoży każdy element przez c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Oblicza wektor, którego i-ty element jest średnią i-tych elementów wektorów wejściowych."""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """Oblicza v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "wektory muszą mieć taką samą długość"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6

def sum_of_squares(v: Vector) -> float:
    """Zwraca v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3

import math

def magnitude(v: Vector) -> float:
    """Zwraca moduł (długość) wektora v"""
    return math.sqrt(sum_of_squares(v))   # Funkcja math.sqrt oblicza wartość pierwiastka kwadratowego.

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    """Oblicza (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Oblicza odległość pomiędzy v i w"""
    return math.sqrt(squared_distance(v, w))


def distance(v: Vector, w: Vector) -> float:  # type: ignore
    return magnitude(subtract(v, w))

# alias typu macierzy
Matrix = List[List[float]]

A = [[1, 2, 3],  # Macierz A ma 2 wiersze i 3 kolumny.
     [4, 5, 6]]

B = [[1, 2],     # Macierz B ma 3 wiersze i 2 kolumny.
     [3, 4],
     [5, 6]]

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Zwraca liczbę wierszy i kolumn macierzy A"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # Liczba elementów pierwszego wiersza.
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 wiersze, 3 kolumny

def get_row(A: Matrix, i: int) -> Vector:
    """Zwraca i-ty wiersz macierzy A (jako wektor)"""
    return A[i]             # A[i] jest już i-tym wierszem.

def get_column(A: Matrix, j: int) -> Vector:
    """Zwraca j-tą kolumnę macierzy A (jako wektor)"""
    return [A_i[j]          # j-ty elementy wiersza A_i.
            for A_i in A]   # Dla każdego wiersza  A_i.

from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Zwraca macierz o wymiarach num_rows x num_cols, 
    której element (i, j) jest definiowany jako entry_fn(i, j).
    """
    return [[entry_fn(i, j)             # Na podstawie danego i utwórz listę
             for j in range(num_cols)]  # [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # Utwórz po jednej liście dla każdego i

def identity_matrix(n: int) -> Matrix:
    """Zwraca macierz jednostkową n x n"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # użytkownik  0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # użytkownik  1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # użytkownik  2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # użytkownik  3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # użytkownik  4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # użytkownik  5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # użytkownik  6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # użytkownik  7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # użytkownik  8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # użytkownik  9

assert friend_matrix[0][2] == 1, "0 i 2 są połączone"
assert friend_matrix[0][8] == 0, "0 i 8 nie są połączone"

# Wystarczy sprawdzić zawartośćjednego wiersza.
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]

