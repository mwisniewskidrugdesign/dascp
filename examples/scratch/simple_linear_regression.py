def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    Błąd predykcji beta * x_i + alpha
    przy rzeczywistej wartości y_i.
    """
    return predict(alpha, beta, x_i) - y_i

from scratch.linear_algebra import Vector

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

from typing import Tuple
from scratch.linear_algebra import Vector
from scratch.statistics import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Na podstawie przekazanych wartości treningowych x i y
    znajdź za pomocą metody najmniejszych kwadratów optymalne wartości alpha i beta.
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

# Powinno znaleźć y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)

from scratch.statistics import num_friends_good, daily_minutes_good

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

from scratch.statistics import de_mean

def total_sum_of_squares(y: Vector) -> float:
    """Suma odchyleń kwadratów wartości y_i od średniej."""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    Ułamek wariancji y wychwycony przez model równy
    1 - ułamek wariancji y niewychwycony przez model.
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330

def main():
    import random
    import tqdm
    from scratch.gradient_descent import gradient_step
    
    num_epochs = 10000
    random.seed(0)
    
    guess = [random.random(), random.random()]  # na początek wybierz wartość losową
    
    learning_rate = 0.00001
    
    with tqdm.trange(num_epochs) as t:
        for _ in t:
            alpha, beta = guess
    
            # Pochodna cząstkowa straty w odniesieniu do alpha 
            grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                         for x_i, y_i in zip(num_friends_good,
                                             daily_minutes_good))
    
            # Pochodna cząstkowa straty w odniesieniu do beta
            grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                         for x_i, y_i in zip(num_friends_good,
                                             daily_minutes_good))
    
            # Obliczamy stratę, aby wstawić do opisu tqdm
            loss = sum_of_sqerrors(alpha, beta,
                                   num_friends_good, daily_minutes_good)
            t.set_description(f"loss: {loss:.3f}")
    
            # Na koniec zaktualizuj przewidywanie
            guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
    
    # Powinniśmy otrzymać mniej więcej taki sam wynik:
    alpha, beta = guess
    assert 22.9 < alpha < 23.0
    assert 0.9 < beta < 0.905
    
if __name__ == "__main__": main()
