# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random

def sum_of_squares(v):
    """Oblicza sumę elementów obiektu v podniesionych do kwadratu."""
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def plot_estimated_derivative():

    def square(x):
        return x * x

    def derivative(x):
        return 2 * x

    derivative_estimate = lambda x: difference_quotient(square, x, h=0.00001)

    # Wykres pokazuje, że uzyskane wartości są praktycznie identyczne.
    import matplotlib.pyplot as plt
    x = range(-10,10)
    plt.plot(x, map(derivative, x), 'rx')           # czerwony  x
    plt.plot(x, map(derivative_estimate, x), 'b+')  # niebieski +
    plt.show()                                      # purple *, hopefully

def partial_difference_quotient(f, v, i, h):

    # Dodaj h tylko do i-tego elementu wektora v.
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
         
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)] 

def step(v, direction, step_size):
    """Przejdź o step_size w kierunku v."""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v): 
    return [2 * v_i for v_i in v]

def safe(f):
    """Zwraca nową funkcję taką samą jak f, ale zwracającą nieskończoność w przypadku wystąpienia błędu."""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # Symbol nieskończoności w Pythonie
    return safe_f


#
# 
# minimize / maximize batch
#
#

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """Uzyj metody najmniejszego gradientu w celu określenia wartości theta minimalizujących funkcję celu."""
    
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    theta = theta_0                           # Przypisz początkową wartość theta.
    target_fn = safe(target_fn)               # Bezpieczna wersja funkcji target_fn.
    value = target_fn(theta)                  # Minimalizowana wartość.
    
    while True:
        gradient = gradient_fn(theta)  
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]
                   
        # Wybierz wartość minimalizującą funkcję błędu.       
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        
        # Zatrzymaj w przypadku osiągnięcia punktu zbieżności.
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    """Zwróć funkcję, która dla dowolnego x zwraca -f(x)."""
    return lambda *args, **kwargs: -f(*args, **kwargs)
    
def negate_all(f):
    """Zrób to samo w przypadku, gdy f zwraca listę liczb."""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0, 
                          tolerance)

#
# Stochastyczna metoda gradientu prostego
#

def in_random_order(data):
    """Generator zwracający elementy zbioru danych w losowej kolejności."""
    indexes = [i for i, _ in enumerate(data)]  # Utwórz listę indeksów.
    random.shuffle(indexes)                    # Zmień kolejność indeksów.
    for i in indexes:                          # Zwróć dane w tej kolejności.
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = zip(x, y)
    theta = theta_0                             # Punkt początkowy.
    alpha = alpha_0                             # Początkowy rozmiar kroku.
    min_theta, min_value = None, float("inf")   # Dotychczasowe minimum.
    iterations_with_no_improvement = 0
    
    # Zatrzymaj algorytm po 100 iteracjach bez poprawy.
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # W przypadku znalezienia nowego minimum zapamiętaj je
            # i wróć do początkowego rozmiaru kroku.
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # W przypadku braku poprawy zmniejsz rozmiar kroku.
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # Wykonaj krok gradientu dla każdego elementu zbioru.
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)

if __name__ == "__main__":

    print "using the gradient"

    v = [random.randint(-10,10) for i in range(3)]

    tolerance = 0.0000001

    while True:
        # v, sum_of_squares(v)
        gradient = sum_of_squares_gradient(v)   # Oblicz gradient v.
        next_v = step(v, gradient, -0.01)       # Wykonaj krok w kierunku przeciwnym do gradientu.
        if distance(next_v, v) < tolerance:     # Zatrzymaj w przypadku osiągnięcia zbieżności.
            break
        v = next_v                              # Kontynuuj w przeciwnym wypadku.

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
    print


    print "using minimize_batch"

    v = [random.randint(-10,10) for i in range(3)]

    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
