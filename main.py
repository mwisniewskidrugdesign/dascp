import math
import matplotlib.pyplot as plt

# jeśli niezależne to prawdopodobieństwo wystąpienia zdarzenia E i F równa się iloczyn prawdopodobieństw każdego z nich
# jeśli zdazenia są niezależne to wystąpienie P(E|F) = P(E)
# jeśli zdarzenia są zależne to wystąpienie P(E|F) = P(E∩F)/P(F)


## PRZYKŁAD Z RODZINĄ ##

gender = ['girl','boy']

# D - obydwoje dzieci jest płci żeńskiej
# S - starsze dziecko jest płci żeńskiej

# P(D|S) - obydwoje dzieci jest płci żeńskiej jeśli starsze dziecko było płci żeńskiej
# P(D|S) = P(D∩S)/P(S) = P(D) / P(S)


# P(J) - przynajmniej jedno dziecko jest dziewczynką
# P(D|J)  oboje dzieci jest płci żeńskiej pod warunkiem że jedno z nich jest płci żeńskiej

import enum, random

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0



for _ in range(10000):
    younger = random_kid()  #wybiera 0 lub 1
    older = random_kid()    #wybiera 0 lub 1
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print('P(obydwoje | starsze): ', both_girls / older_girl)
print('P(obydwoje | którekolwiek)', both_girls / either_girl)

### ROZKŁAD NORMALNY

SQRT_TWO_PI = math.sqrt(2*math.pi)
def normal_pdf(x: float, mi: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mi)**2 / 2 / sigma**2)) / (SQRT_TWO_PI * sigma)


def bernoulli_trial(p: float) -> int:
    """Zwraca 1 z prawdopodobienstwem p i 0 z prawdopodobienstwem 1 - p"""
    return 1 if random.random() < p else 0

def binominal(n: int, p: float) -> int:
    """zwraca sumę z prób Bernoulliego"""
    return sum(bernoulli_trial(p) for _ in range(n))

a = binominal(10000000, 0.55)
print(a)