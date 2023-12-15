def uniform_cdf(x: float) -> float:
    """Zwraca prawdopodobieństwo tego, że zmienna rozkładu jednostajnego jest <= x."""
    if x < 0:   return 0    # Rozkład jednostajny nigdy nie przyjmuje wartości mniejszych od 0,
    elif x < 1: return x    # np. P(X <= 0,4) = 0,4.
    else:       return 1    # Liczba wylosowana z rozkładu jednostajnego jest zawsze mniejsza od 1.

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mi: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mi) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mi=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mi=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mi=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mi=-1)   for x in xs],'-.',label='mi=-1,sigma=1')
plt.legend()
plt.title("Wykresy roznych rozkladow normalnych")
plt.show()


plt.savefig('im/various_normal_pdfs.png')
plt.gca().clear()
plt.close()
plt.clf()

def normal_cdf(x: float, mi: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mi) / math.sqrt(2) / sigma)) / 2

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mi=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mi=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mi=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mi=-1) for x in xs],'-.',label='mi=-1,sigma=1')
plt.legend(loc=4) # prawy dolny róg
plt.title("Dystrybuanty roznych rozkladow normalnych")
plt.show()


plt.close()
plt.gca().clear()
plt.clf()

def inverse_normal_cdf(p: float,
                       mi: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
    """Znajduje przybliżoną wartość odwrotności przy użyciu algorytmu wyszukiwania binarnego."""

    # Jeżeli rozkład nie jest standardowy, to oblicz jego standardową postać i przeskaluj.
    if mi != 0 or sigma != 1:
        return mi + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0                      # normal_cdf(-10) ma wartość (zbliżoną do) 0
    hi_z  =  10.0                      # normal_cdf(10)  ma wartość (zbliżoną do) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # Weź pod uwagę punkt środkowy
        mid_p = normal_cdf(mid_z)      # i znajdującą się tam wartość dystrybuanty.
        if mid_p < p:
            low_z = mid_z              # Punkt środkowy znajduje się za nisko, szukaj nad nim.
        else:
            hi_z = mid_z               # Punkt środkowy znajduje się za wysoko, szukaj pod nim.

    return mid_z


import random

def bernoulli_trial(p: float) -> int:
    """Zwraca 1 z prawdopodobieństwem p i 0 z prawdopodobieństwem 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Zwraca sumę n prób Bernoulliego(p)"""
    return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def make_hist(p: float, n: int, num_points: int) -> None:
    """Rysuje histogram punktów z dwumianu(n, p)"""
    data = [binomial(n, p) for _ in range(num_points)]

    # Próbki dwumianu przedstaw na wykresie słupkowym.
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mi = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # Przybliżenie rozkładu normalnego przedstaw na wykresie liniowym
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mi, sigma) - normal_cdf(i - 0.5, mi, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.title("Rozklad dwumianu a przyblizenie rozkladu normalnego")
    plt.show()

def main():
    import enum, random
    
    # Enum jest zbiorem wartości typu enumerated. 
    # Możemy użyć ich, aby nasz kod był bardziej czytelny.
    class Kid(enum.Enum):
        BOY = 0
        GIRL = 1
    
    def random_kid() -> Kid:
        return random.choice([Kid.BOY, Kid.GIRL])
    
    both_girls = 0
    older_girl = 0
    either_girl = 0
    
    random.seed(0)
    
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == Kid.GIRL:
            older_girl += 1
        if older == Kid.GIRL and younger == Kid.GIRL:
            both_girls += 1
        if older == Kid.GIRL or younger == Kid.GIRL:
            either_girl += 1
    
    print("P(obydwoje | starsze):", both_girls / older_girl)     # 0.514 ~ 1/2
    print("P(obydwoje | którekolwiek): ", both_girls / either_girl)  # 0.342 ~ 1/3
    
    
    
    assert 0.48 < both_girls / older_girl < 0.52
    assert 0.30 < both_girls / either_girl < 0.35
    
    def uniform_pdf(x: float) -> float:
        return 1 if 0 <= x < 1 else 0
    
if __name__ == "__main__": main()
