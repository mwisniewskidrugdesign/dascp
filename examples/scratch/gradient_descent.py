from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """Oblicza sumę elementów obiektu v podniesionych do kwadratu."""
    return dot(v, v)

from typing import Callable

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Przejdź o step_size w kierunku gradient od punktu v."""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# x jest z zakresu od –50 do 49, y wynosi zawsze 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # Przewidywanie modelu.
    error = (predicted - y)              # Błąd obliczamy jako (predicted – actual).
    squared_error = error ** 2           # Minimalizujemy błąd kwadratowy,
    grad = [2 * error * x, 2 * error]    # używając jego gradientu.
    return grad

from typing import TypeVar, List, Iterator

T = TypeVar('T')  # # dzięki temu możemy pisać funkcje generyczne (bez określonego typu)

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generuje próbkę ze zbioru danych o rozmiarze batch_size"""
    # start przyjmuje wartości: 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # pomieszaj podzbiory

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

def main():
    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimates = [difference_quotient(square, x, h=0.001) for x in xs]
    
    # Wykres pokazuje, że uzyskane wartości są praktycznie identyczne.
    import matplotlib.pyplot as plt
    plt.title("Actual Derivatives vs. Estimates")
    plt.plot(xs, actuals, 'rx', label='Actual')       # czerwony x
    plt.plot(xs, estimates, 'b+', label='Estimate')   # niebieski +
    plt.legend(loc=9)
    plt.show()
    
    
    plt.close()
    
    def partial_difference_quotient(f: Callable[[Vector], float],
                                    v: Vector,
                                    i: int,
                                    h: float) -> float:
        """Oblicz i-ty iloraz różnicowy pochodnej cząstkowej f wektora v."""
        w = [v_j + (h if j == i else 0)    # Dodaj h tylko do i-tego elementu wektora v.             for j, v_j in enumerate(v)]
    
        return (f(w) - f(v)) / h
    
    
    # Podrozdział "Korzystanie z gradientu" 
    
    # Wybierz losowy punkt początkowy.
    v = [random.uniform(-10, 10) for i in range(3)]
    
    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)    # Oblicz gradient  w punkcie v.
        v = gradient_step(v, grad, -0.01)    # tWykonaj krok w kierunku przeciwnym do gradientu
        print(epoch, v)
    
    assert distance(v, [0, 0, 0]) < 0.001    # v powinno być zbliżone do 0.
    
    
    # Pierwszy przykład z podrozdziału "Używanie metody gradientu do dopasowywania modeli"
    
    from scratch.linear_algebra import vector_mean
    
    # Rozpoczynamy od losowych wartości nachylenia (slope) i wyrazu wolnego (intercept) funkcji liniowej
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    learning_rate = 0.001
    
    for epoch in range(5000):
        # Obliczamy średnią gradientów
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        # Robimy krok w tym kierunku
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "slope should be about 20"
    assert 4.9 < intercept < 5.1, "intercept should be about 5"
    
    
    # Przykład: minibatch
    
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "wartość slope powinna wynosić około 20"
    assert 4.9 < intercept < 5.1, "wartość intercept powinna wynosić około 5"
    
    
    # Przykład: metoda stochastyczna
    
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta
    assert 19.9 < slope < 20.1,   "wartość slope powinna wynosić około 20"
    assert 4.9 < intercept < 5.1, "wartość intercept powinna wynosić około 5"
    
if __name__ == "__main__": main()
