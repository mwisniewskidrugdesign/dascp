from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Określa wartości mi i sigma."""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

from scratch.probability import normal_cdf

# Funkcja dystrybuanty rozkładu normalnego określa prawdopodobieństwo tego, że zmienna znajduje się poniżej wartości progowej.
normal_probability_below = normal_cdf

# Jeżeli nie znajduje się nad wartością progową, to znajduje się pod nią.
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """Prawdopodobieństwo tego, że N(mi, sigma) jest większe niż lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# Wartość znajduje się w przedziale, jeżeli jest mniejsza od górnej wartości granicznej i większa od dolnej wartości granicznej.
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """Prawdopodobieństwo tego, że N(mi, sigma) jest pomiędzy lo i hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# Wartość jest poza przedziałem, jeżeli nie znajduje się pomiędzy ograniczeniami.
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """Prawdopodobieństwo tego, że N(mi, sigma) nie jest pomiędzy lo i hi."""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Zwraca z przy zachowaniu warunku P(Z <= z) = prawdopodobieństwo."""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Zwraca z przy zachowaniu warunku P(Z >= z) = prawdopodobieństwo"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    Zwraca granice symetryczne (umieszczone wokół średniej), 
    które obejmują określone prawdopodobieństwo.
    """
    tail_probability = (1 - probability) / 2

    # Nad górną granicą powinna znajdować się wartość tail_probability.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # Pod dolną granicą powinna znajdować się wartość tail_probability.
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)


assert mu_0 == 500
assert 15.8 < sigma_0 < 15.9

# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)


assert 468.5 < lower_bound < 469.5
assert 530.5 < upper_bound < 531.5

# Ograniczenie 95% na podstawie założenia, że p jest równe 0,5.
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# Rzeczywiste wartości mi i sigma przy założeniu, że p = 0,55.
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# Błąd typu drugiego oznacza błąd polegający na nieodrzuceniu hipotezy zerowej,
# do czego dochodzi, gdy X wciąż znajduje się w naszym początkowym interwale.
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.887


assert 0.886 < power < 0.888

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# Wynosi 526 (< 531, ponieważ potrzebujemy więcej prawdopodobieństwa w górnej części).

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.936


assert 526 < hi < 526.1
assert 0.9363 < power < 0.9364

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Jak prawdopodobne jest zaobserwowanie wartości przynajmniej tak skrajnej jak x
    (w obydwu kierunkach) jeżeli wartości są z przedziału N(mi, sigma)?
    """
    if x >= mu:
        # Jeżeli x jest większy od średniej…
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # Jeżeli x jest mniejszy od średniej….
        return 2 * normal_probability_below(x, mu, sigma)

two_sided_p_value(529.5, mu_0, sigma_0)   # 0.062

import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0    # Policz liczbę wyrzuconych orłów
                    for _ in range(1000))                # podczas 1000 rzutów monetą.
    if num_heads >= 530 or num_heads <= 470:             # Oblicz, ile razy liczba ta
        extreme_value_count += 1                         # osiąga „ekstremum”.

# wartość p wynosi 0,062, więc około 62 skrajne wartości z 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

two_sided_p_value(531.5, mu_0, sigma_0)   # 0.0463


tspv = two_sided_p_value(531.5, mu_0, sigma_0)
assert 0.0463 < tspv < 0.0464

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_p_value(524.5, mu_0, sigma_0) # 0.061

upper_p_value(526.5, mu_0, sigma_0) # 0.047

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)   # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)        # [0.4940, 0.5560]

p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709]

from typing import List

def run_experiment() -> List[bool]:
    """Wykonaj 1000 rzutów monetą. True oznacza orła, a False reszkę."""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """Zakłada 5% poziom ufności."""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

assert num_rejections == 46

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

z = a_b_test_statistic(1000, 200, 1000, 180)    # -1.14


assert -1.15 < z < -1.13

two_sided_p_value(z)                            # 0.254


assert 0.253 < two_sided_p_value(z) < 0.255

z = a_b_test_statistic(1000, 200, 1000, 150)    # -2.94
two_sided_p_value(z)                            # 0.003

def B(alpha: float, beta: float) -> float:
    """Stała normalizacji zapewniająca całkowite prawdopodobieństwo równe 1."""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:          # Wszystkie wagi mieszczą się w zakresie [0, 1].
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

