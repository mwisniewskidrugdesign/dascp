from typing import List
from typing import Tuple
from typing import Tuple
from typing import Callable

import math

Vector = List[float]
Matrix = List[List[float]]

def add(v: Vector, w: Vector) -> Vector:
    """dodawanie wektorów"""
    assert len(v) == len(w), "wektory muszą mieć tę samą długość"

    return [v_i + w_i for v_i, w_i in zip(v, w)]
def subtract(v: Vector, w: Vector) -> Vector:
    """odejmowanie wektorów"""
    assert len(v) == len(w), "wektory muszą mieć tę samą długość "

    return [v_i - w_i for v_i, w_i in zip(v, w)]
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
def scalar_multiply(c: float, v: Vector) -> Vector:
    """Mnoży każdy element przez c"""
    return [c * v_i for v_i in v]
def vector_mean(vectors: List[Vector]) -> Vector:
    """Oblicza wektor, którego i-ty element jest średnią i-tych elementów wektorów wejściowych."""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
def dot(v: Vector, w: Vector) -> float:
    """Oblicza v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "wektory muszą mieć taką samą długość"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
def sum_of_squares(v: Vector) -> float:
    """Zwraca v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)
def magnitude(v: Vector) -> float:
    """Zwraca moduł (długość) wektora v"""
    return math.sqrt(sum_of_squares(v))   # Funkcja math.sqrt oblicza wartość pierwiastka kwadratowego.
def squared_distance(v: Vector, w: Vector) -> float:
    """Oblicza (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))
def distance(v: Vector, w: Vector) -> float:  # type: ignore
    return magnitude(subtract(v, w))
def shape(A: Matrix) -> Tuple[int, int]:
    """Zwraca liczbę wierszy i kolumn macierzy A"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # Liczba elementów pierwszego wiersza.
    return num_rows, num_cols
def get_row(A: Matrix, i: int) -> Vector:
    """Zwraca i-ty wiersz macierzy A (jako wektor)"""
    return A[i]             # A[i] jest już i-tym wierszem.
def get_column(A: Matrix, j: int) -> Vector:
    """Zwraca j-tą kolumnę macierzy A (jako wektor)"""
    return [A_i[j]          # j-ty elementy wiersza A_i.
            for A_i in A]   # Dla każdego wiersza  A_i.
def make_matrix(num_rows: int,num_cols: int,entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j)
             for j in range(num_cols)]  # [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # Utwórz po jednej liście dla każdego i
def identity_matrix(n: int) -> Matrix:
    """Zwraca macierz jednostkową n x n"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

