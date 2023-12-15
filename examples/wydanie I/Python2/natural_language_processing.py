# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import math, random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests

def plot_resumes(plt):
    data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

    def text_size(total):
        """Wynosi 8, jeżeli liczba wystąpień jest równa 0, lub 28, jeżeli liczba wystąpień jest równa 200."""
        return 8 + total / 200 * 20

    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))
    plt.xlabel("Popularnosc w ofertach pracy")
    plt.ylabel("Popularnosc w CV")
    plt.axis([0, 100, 0, 100])
    plt.show()

#
# n-gram models
#

def fix_unicode(text):
    return text.replace(u"\u2019", "'")

def get_document():

    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')

    content = soup.find("div", "article-body")        # Znajdź element div oznaczony etykietą entry-content.
    regex = r"[\w']+|[\.]"                            # Wybiera słowa i kropki.

    document = []


    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document

def generate_using_bigrams(transitions):
    current = "."   # Kolejne słowo rozpoczyna nowe zdanie.
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigramy (current, _)
        current = random.choice(next_word_candidates)  # Wylosuj.
        result.append(current)                         # Dołącz do wyniku.
        if current == ".": return " ".join(result)     # Zakończ pracę w przypadku kropki.

def generate_using_trigrams(starts, trigram_transitions):
    current = random.choice(starts)   # Wylosuj pierwsze słowo.
    prev = "."                        # Umieść przed nim kropkę.
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next = random.choice(next_word_candidates)

        prev, current = current, next
        result.append(current)

        if current == ".":
            return " ".join(result)

def is_terminal(token):
    return token[0] != "_"

def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        # Ignoruj węzły końcowe.
        if is_terminal(token): continue

        # Wylosuj element zastępujący (znaleziono znacznik niekońcowy).
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        return expand(grammar, tokens)

    # W tym momencie wszystkie węzły zostały obsłużone.
    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])

#
# Gibbs Sampling
#

def roll_a_die():
    return random.choice([1,2,3,4,5,6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_given_x(x):
    """Może wynosić x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()

def random_x_given_y(y):
    if y <= 7:
        # Jeżeli suma jest równa 7 lub mniej, to na pierwszej kostce
        # mogły wypaść wartości 1, 2, ..., (suma - 1).
        return random.randrange(1, y)
    else:
        # Jeżeli suma jest większa od 7, to na pierwszej kostce
        # mogły równie prawdopodobnie wypaść wartości (suma - 6), (suma - 5), ..., 6.
        return random.randrange(y - 6, 7)

def gibbs_sample(num_iters=100):
    x, y = 1, 2 # To tak naprawdę nie ma znaczenia.
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def compare_distributions(num_samples=1000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

#
# Modelowanie tematu
#

def sample_from(weights):
    total = sum(weights)
    rnd = total * random.random()       # Rozkład jednostajny od 0 do sumy.
    for i, w in enumerate(weights):
        rnd -= w                        # Zwraca najmniejszą wartość i spełniającą warunek:
        if rnd <= 0: return i           # sum(weights[:(i+1)]) >= rnd.

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4

document_topic_counts = [Counter()
                         for _ in documents]

topic_word_counts = [Counter() for _ in range(K)]

topic_counts = [0 for _ in range(K)]

document_lengths = map(len, documents)

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

D = len(documents)

def p_topic_given_document(topic, d, alpha=0.1):
    """Ułamek słów w dokumencie _d_, które są przypisane
    do tematu _topic_ plus wartość wygładzająca."""

    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word, topic, beta=0.1):
    """Ułamek słów _word_ przypisanych do tematu _topic_,
    plus wartość wygładzająca."""

    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))

def topic_weight(d, word, k):
    """Zwróć wagę k-tego tematu na podstawie
    dokumentu i występującego w nim słowa"""

    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

for iter in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):

            # Odejmij parę słowo-temat od sumy,
            # aby nie wpływała ona na wagi.
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # Wybierz nowy temat na podstawie wag.
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            # Dodaj go z powrotem do sumy.
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

if __name__ == "__main__":

    document = get_document()

    bigrams = zip(document, document[1:])
    transitions = defaultdict(list)
    for prev, current in bigrams:
        transitions[prev].append(current)

    random.seed(0)
    print "bigram sentences"
    for i in range(10):
        print i, generate_using_bigrams(transitions)
    print

    # trigramy

    trigrams = zip(document, document[1:], document[2:])
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in trigrams:

        if prev == ".":              # Jeżeli poprzednim „słowem” była kropka,
            starts.append(current)   # to  zacznij od zacznij słowa

        trigram_transitions[(prev, current)].append(next)

    print "trigram sentences"
    for i in range(10):
        print i, generate_using_trigrams(starts, trigram_transitions)
    print

    grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big", "linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]
    }

    print "grammar sentences"
    for i in range(10):
        print i, " ".join(generate_sentence(grammar))
    print

    print "gibbs sampling"
    comparison = compare_distributions()
    for roll, (gibbs, direct) in comparison.iteritems():
        print roll, gibbs, direct


    # topic MODELING

    for k, word_counts in enumerate(topic_word_counts):
        for word, count in word_counts.most_common():
            if count > 0: print k, word, count

    topic_names = ["Big Data and programming languages",
                   "Python and statistics",
                   "databases",
                   "machine learning"]

    for document, topic_counts in zip(documents, document_topic_counts):
        print document
        for topic, count in topic_counts.most_common():
            if count > 0:
                print topic_names[topic], count,
        print
