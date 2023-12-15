# -*- coding: utf-8 -*-
from __future__ import division
from probability import normal_cdf, inverse_normal_cdf
import math, random

def normal_approximation_to_binomial(n, p):
    """Określa wartości mu i sigma."""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

#####
#
# probabilities a normal lies in an interval
#
######

# Funkcja cdf rozkładu normalnego określa prawdopodobieństwo tego, że zmienna znajduje się poniżej wartości progowej.
normal_probability_below = normal_cdf

# Jeżeli nie znajduje się nad wartością progową, to znajduje się pod nią.
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)
    
# Wartość znajduje się w przedziale, jeżeli 
# jest mniejsza od górnej wartości granicznej i większa od dolnej wartości granicznej.
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# Wartość jest poza przedziałem jeżeli nie znajduje się pomiędzy ograniczeniami.
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

######
#
#  normal bounds
#
######


def normal_upper_bound(probability, mu=0, sigma=1):
    """Zwraca z przy zachowaniu warunku P(Z <= z) = prawdopodobieństwo."""
    return inverse_normal_cdf(probability, mu, sigma)
    
def normal_lower_bound(probability, mu=0, sigma=1):
    """Zwraca z przy zachowaniu warunku P(Z >= z) = prawdopodobieństwo"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """Zwraca granice symetryczne (umieszczone wokół średniej), 
    które obejmują określone prawdopodobieństwo."""
    tail_probability = (1 - probability) / 2

    # Nad górną granicą powinna znajdować się wartość tail_probability.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # Pod dolną granicą powinna znajdować się wartość tail_probability.
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # Jeżeli x jest większy od średniej...
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # Jeżeli x jest mniejszy od średniej...
        return 2 * normal_probability_below(x, mu, sigma)   

def count_extreme_values():
    extreme_value_count = 0
    for _ in range(100000):
        num_heads = sum(1 if random.random() < 0.5 else 0    # Policz liczbę wyrzuconych orłów
                        for _ in range(1000))                # podczas 1000 rzutów monetą.
        if num_heads >= 530 or num_heads <= 470:             # Oblicz ile razy liczba ta
            extreme_value_count += 1                         # osiąga „ekstremum”.

    return extreme_value_count / 100000

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below    

##
#
# P-hacking
#
##

def run_experiment():
    """Wykonaj 1000 rzutów monetą. True oznacza orła, a False reszkę."""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment):
    """Zakłada 5% poziom ufności."""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531


##
#
# running an A/B test
#
##

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

##
#
# Wnioskowanie bayesowskie
#
##

def B(alpha, beta):
    """Stała normalizacji zapewniajaca całkowite prawdopodobieństwo równe 1."""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:          # Wszystkie wagi mieszczą się w zakresie [0, 1].    
        return 0        
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


if __name__ == "__main__":

    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds(0.95, mu_0, sigma_0)", normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print
    print "power of a test"
    
    print "Ograniczenie 95% na podstawie założenia, że p jest równe 0,5."
    
    lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print "lo", lo
    print "hi", hi

    print "Rzeczywiste wartości mu i sigma przy założeniu, że p = 0,55."
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
    print "mu_1", mu_1
    print "sigma_1", sigma_1

    # Błąd typu drugiego oznacza błąd polegający na nieodrzuceniu hipotezy zerowej,
    # do czego dochodzi, gdy X wciąż znajduje się w naszym początkowym interwale.
    type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
    power = 1 - type_2_probability # 0,887

    print "type 2 probability", type_2_probability
    print "power", power
    print

    print "one-sided test"
    hi = normal_upper_bound(0.95, mu_0, sigma_0) 
    print "hi", hi # Wynosi 256 (< 531, ponieważ potrzebujemy więcej prawdopodobieństwa w górnej części).
    type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type_2_probability # = 0,936
    print "type 2 probability", type_2_probability
    print "power", power
    print

    print "two_sided_p_value(529.5, mu_0, sigma_0)", two_sided_p_value(529.5, mu_0, sigma_0)  

    print "two_sided_p_value(531.5, mu_0, sigma_0)", two_sided_p_value(531.5, mu_0, sigma_0)

    print "upper_p_value(525, mu_0, sigma_0)", upper_p_value(525, mu_0, sigma_0)
    print "upper_p_value(527, mu_0, sigma_0)", upper_p_value(527, mu_0, sigma_0)    
    print 

    print "P-hacking"

    random.seed(0)
    experiments = [run_experiment() for _ in range(1000)]
    num_rejections = len([experiment
                          for experiment in experiments 
                          if reject_fairness(experiment)])

    print num_rejections, "rejections out of 1000"
    print

    print "A/B testing"
    z = a_b_test_statistic(1000, 200, 1000, 180)
    print "a_b_test_statistic(1000, 200, 1000, 180)", z
    print "p-value", two_sided_p_value(z)
    z = a_b_test_statistic(1000, 200, 1000, 150)
    print "a_b_test_statistic(1000, 200, 1000, 150)", z
    print "p-value", two_sided_p_value(z)
