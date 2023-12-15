
import matplotlib.pyplot as plt
plt.gca().clear()

data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]


from matplotlib import pyplot as plt

def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")

import re
from bs4 import BeautifulSoup
import requests

url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

content = soup.find("div", "article-body")   # Znajdź element div oznaczony etykietą article-body.
regex = r"[\w']+|[\.]"                       # Wybiera słowa i kropki.

document = []

for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

from collections import defaultdict

transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
    transitions[prev].append(current)

def generate_using_bigrams() -> str:
    current = "."   # Kolejne słowo rozpoczyna nowe zdanie.
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigramy (current, _)
        current = random.choice(next_word_candidates)  # Wylosuj.
        result.append(current)                         # Dołącz do wyniku.
        if current == ".": return " ".join(result)     # Zakończ pracę w przypadku kropki.

trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in zip(document, document[1:], document[2:]):

    if prev == ".":              # Jeżeli poprzednim „słowem” była kropka,
        starts.append(current)   # to zacznij od tego słowa

    trigram_transitions[(prev, current)].append(next)

def generate_using_trigrams() -> str:
    current = random.choice(starts)   # Wylosuj pierwsze słowo.
    prev = "."                        # Umieść przed nim kropkę.
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result)

from typing import List, Dict

# alias typu
Grammar = Dict[str, List[str]]

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

def is_terminal(token: str) -> bool:
    return token[0] != "_"

def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        # Ignoruj węzły końcowe.
        if is_terminal(token): continue

        # Wylosuj element zastępujący (znaleziono znacznik niekońcowy).
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # Replacement może być np. "_NP _VP", więc musimy podzielić po spacjach
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        # Wywołaj rozszerzanie nowej listy znaczników.
        return expand(grammar, tokens)

    # W tym momencie wszystkie węzły zostały obsłużone.
    return tokens

def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ["_S"])

from typing import Tuple
import random

def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample() -> Tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_given_x(x: int) -> int:
    """Może wynosić x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()

def random_x_given_y(y: int) -> int:
    if y <= 7:
        # Jeżeli suma jest równa 7 lub mniej, to na pierwszej kostce 
        # mogły wypaść wartości 1, 2, ..., (suma - 1).
        return random.randrange(1, y)
    else:
        # Jeżeli suma jest większa od 7, to na pierwszej kostce
        # mogły równie prawdopodobnie wypaść wartości (suma - 6), (suma - 5), ..., 6.
        return random.randrange(y - 6, 7)

def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2 # To tak naprawdę nie ma znaczenia.
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def compare_distributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts

def sample_from(weights: List[float]) -> int:
    """zwraca i z prawdopodobieństwem weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()      # Rozkład jednostajny od 0 do sumy.
    for i, w in enumerate(weights):
        rnd -= w                       # Zwraca najmniejszą wartość i spełniającą warunek:
        if rnd <= 0: return i          # weights[0] + … + weights[i] >= rnd.

from collections import Counter

# wylosuj 1000 razy i policz
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
assert 10 < draws[0] < 190   # powinno być  ~10%
assert 10 < draws[1] < 190   # powinno być  ~10%
assert 650 < draws[2] < 950  # powinno być  ~80%
assert draws[0] + draws[1] + draws[2] == 1000

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

# Lista liczników; po jednym dla każdego dokumentu.
document_topic_counts = [Counter() for _ in documents]

# Lista liczników; po jednym dla każdego tematu.
topic_word_counts = [Counter() for _ in range(K)]

# Lista liczb; po jednej dla każdego tematu.
topic_counts = [0 for _ in range(K)]

# Lista liczb; po jednej dla każdego dokumentu.
document_lengths = [len(document) for document in documents]

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

D = len(documents)

def p_topic_given_document(topic: int, d: int, alpha: float = 0.1) -> float:
    """
    Ułamek słów w dokumencie _d_, które są przypisane
    do tematu _topic_ plus wartość wygładzająca.
    """
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word: str, topic: int, beta: float = 0.1) -> float:
    """
    Ułamek słów _word_ przypisanych do tematu _topic_,
    plus wartość wygładzająca.
    """
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))

def topic_weight(d: int, word: str, k: int) -> float:
    """
    Zwróć wagę k-tego tematu na podstawie
    dokumentu i występującego w nim słowa
    """
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d: int, word: str) -> int:
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

import tqdm

for iter in tqdm.trange(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):

            # Odejmij parę słowo-temat od sumy, aby nie wpływała ona na wagi.
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

for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0:
            print(k, word, count)

topic_names = ["Big Data and programming languages",
               "Python and statistics",
               "databases",
               "machine learning"]

for document, topic_counts in zip(documents, document_topic_counts):
    print(document)
    for topic, count in topic_counts.most_common():
        if count > 0:
            print(topic_names[topic], count)
    print()

from scratch.linear_algebra import dot, Vector
import math

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))

assert cosine_similarity([1., 1, 1], [2., 2, 2]) == 1, "ten sam kierunek"
assert cosine_similarity([-1., -1], [2., 2]) == -1,    "przeciwny kierunek"
assert cosine_similarity([1., 0], [0., 1]) == 0,       "ortogonalne"

colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]

def make_sentence() -> str:
    return " ".join([
        "The",
        random.choice(colors),
        random.choice(nouns),
        random.choice(verbs),
        random.choice(adverbs),
        random.choice(adjectives),
        "."
    ])

NUM_SENTENCES = 50

random.seed(0)
sentences = [make_sentence() for _ in range(NUM_SENTENCES)]

from scratch.deep_learning import Tensor

class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {}  # mapowanie słowa na jego identyfikator
        self.i2w: Dict[int, str] = {}  # mapowanie identyfikatora na słowo

        for word in (words or []):     # jeżeli na wejściu były jakieś słowa, 
            self.add(word)             # to je dodajemy.

    @property
    def size(self) -> int:
        """ile słów jest w słowniku"""
        return len(self.w2i)

    def add(self, word: str) -> None:
        if word not in self.w2i:        # jeżeli słowo jest nowe
            word_id = len(self.w2i)     # znajdź kolejny identyfikator.
            self.w2i[word] = word_id    # dodaj do mapowania słów na identyfikatory.
            self.i2w[word_id] = word    # dodaj do mapowania identyfikatorów na słowa.

    def get_id(self, word: str) -> int:
        """zwraca identyfikator słowa (lub None)"""
        return self.w2i.get(word)

    def get_word(self, word_id: int) -> str:
        """zwraca słowo o określonym identyfikatorze (lub None)"""
        return self.i2w.get(word_id)

    def one_hot_encode(self, word: str) -> Tensor:
        word_id = self.get_id(word)
        assert word_id is not None, f"nieznane słowo {word}"

        return [1.0 if i == word_id else 0.0 for i in range(self.size)]

vocab = Vocabulary(["a", "b", "c"])
assert vocab.size == 3,              "w słowniku są 3 słowa"
assert vocab.get_id("b") == 1,       "b powinno mieć identyfikator 1"
assert vocab.one_hot_encode("b") == [0, 1, 0]
assert vocab.get_id("z") is None,    "w słowniku nie ma z"
assert vocab.get_word(2) == "c",     "identyfikator 2 powinien oznaczać c"
vocab.add("z")
assert vocab.size == 4,              "teraz w słowniku są 4 słowa"
assert vocab.get_id("z") == 3,       "teraz z powinno mieć identyfikator 3"
assert vocab.one_hot_encode("z") == [0, 0, 0, 1]

import json

def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(vocab.w2i, f)       # potrzebujemy zapisać jedynie w2i

def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename) as f:
        # odczytaj w2i i na jego podstawie wygeneruj i2w 
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
    return vocab

from typing import Iterable
from scratch.deep_learning import Layer, Tensor, random_tensor, zeros_like

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # jeden wektor o rozmiarze size embedding_dim dla każdego przekształcenia
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zeros_like(self.embeddings)

        # zapisz ostatni input id
        self.last_input_id = None

    def forward(self, input_id: int) -> Tensor:
        """Wybiera wektor odpowiadający wejściowemu identyfikatorowi"""
        self.input_id = input_id    # zapamiętaj, aby użyć przy propagacji wstecznej

        return self.embeddings[input_id]

    def backward(self, gradient: Tensor) -> None:
        # korzystamy z gradientu z poprzedniego przetwarzania
        # jest to znacznie szybsze niż tworzenie zerowego tensora za każdym razem.
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row

        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient

    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]

    def grads(self) -> Iterable[Tensor]:
        return [self.grad]

class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        # wywołaj konstruktor klasy Embedding
        super().__init__(vocab.size, embedding_dim)

        # umieść słownik w obiekcie klasy
        self.vocab = vocab

    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None

    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        """Zwraca n najbliższych słów na podstawie podobieństwa kosinusowego"""
        vector = self[word]

        # wyznacz pary (similarity, other_word) i posortuj według największego podobieństwa
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)

        return scores[:n]

from scratch.deep_learning import tensor_apply, tanh

class SimpleRnn(Layer):
    """Praktycznie najprostsza warstwa rekurencyjna."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w = random_tensor(hidden_dim, input_dim, init='xavier')
        self.u = random_tensor(hidden_dim, hidden_dim, init='xavier')
        self.b = random_tensor(hidden_dim)

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]

    def forward(self, input: Tensor) -> Tensor:
        self.input = input              # zachowaj zarówno wartość wejściową, jak i poprzedni
        self.prev_hidden = self.hidden  # stan ukryty, aby użyć ich w propagacji wstecznej.

        a = [(dot(self.w[h], input) +           # wagi wejściowe
              dot(self.u[h], self.hidden) +     # wagi stanu ukrytego
              self.b[h])                        # wartość progowa
             for h in range(self.hidden_dim)]

        self.hidden = tensor_apply(tanh, a)  # Zastosuj tanh jako funkcję aktywacji
        return self.hidden                   # i zwróć wynik.

    def backward(self, gradient: Tensor):
        # propagacja wsteczna przez funkcję tanh
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]

        # b ma ten sam gradient co a
        self.b_grad = a_grad

        # Każda wartość w[h][i] jest mnożona przez input[i] i dodawana do a[h],
        # więc każdy w_grad[h][i] = a_grad[h] * input[i]
        self.w_grad = [[a_grad[h] * self.input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]

        # Każda wartość u[h][h2] jest mnożona przez hidden[h2] i dodawana do a[h],
        # więc każdy  u_grad[h][h2] = a_grad[h] * prev_hidden[h2]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]

        # Każda wartość input[i] jest mnożona przez każdą wartość w[h][i] i dodawana do a[h],
        # więc każdy input_grad[i] = sum(a_grad[h] * w[h][i] for h in …)
        return [sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]

def main():
    from matplotlib import pyplot as plt
    
    def text_size(total: int) -> float:
        """
        Wynosi 8, jeżeli liczba wystąpień jest równa 0
        lub 28, jeżeli liczba wystąpień jest równa 200.
        """
        return 8 + total / 200 * 20
    
    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word,
                 ha='center', va='center',
                 size=text_size(job_popularity + resume_popularity))
    plt.xlabel("Popularnosc w ofertach pracy")
    plt.ylabel("Popularnosc w CV")
    plt.axis([0, 100, 0, 100])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
    plt.close()
    
    import re
    
    # To wyrażenie regularne nie jest szczególnie zaawansowane, ale w naszym przypadku wystarczy.
    tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower())
                           for sentence in sentences]
    
    # Tworzenie słownika (czyli mapowania słowo -> identyfikator słowa) na podstawie tekstu.
    vocab = Vocabulary(word
                       for sentence_words in tokenized_sentences
                       for word in sentence_words)
    
    from scratch.deep_learning import Tensor, one_hot_encode
    
    inputs: List[int] = []
    targets: List[Tensor] = []
    
    for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):          # dla każdego słowa
            for j in [i - 2, i - 1, i + 1, i + 2]:   # weź najbliższe otoczenie
                if 0 <= j < len(sentence):           # które znajduje się w tekście
                    nearby_word = sentence[j]        # i pobierz z niego słowa.
    
                    # dodaje input, czyli identyfikator word_id pierwotnego słowa.
                    inputs.append(vocab.get_id(word))
    
                    # dodaje target, czyli identyfikatory najbliższych słów.
                    targets.append(vocab.one_hot_encode(nearby_word))
    
    
    # Model uczący dla wektorów słów.
    
    from scratch.deep_learning import Sequential, Linear
    
    random.seed(0)
    EMBEDDING_DIM = 5  # wydaje się być dobrą wielkością
    
    # tworzymy warstwę embedding osobno.
    embedding = TextEmbedding(vocab=vocab, embedding_dim=EMBEDDING_DIM)
    
    model = Sequential([
        # Mając słowo na wejściu (jako wektor identyfikatorów word_ids), dołącz jego wektor.
        embedding,
        # użyj warstwy linear do obliczenia wartości dla najbliższych słów.
        Linear(input_dim=EMBEDDING_DIM, output_dim=vocab.size)
    ])
    
    
    # Trenowanie modelu wektorów słów
    
    from scratch.deep_learning import SoftmaxCrossEntropy, Momentum, GradientDescent
    
    loss = SoftmaxCrossEntropy()
    optimizer = GradientDescent(learning_rate=0.01)
    
    for epoch in range(100):
        epoch_loss = 0.0
        for input, target in zip(inputs, targets):
            predicted = model.forward(input)
            epoch_loss += loss.loss(predicted, target)
            gradient = loss.gradient(predicted, target)
            model.backward(gradient)
            optimizer.step(model)
        print(epoch, epoch_loss)            # wyświetl wartość straty
        print(embedding.closest("black"))   # oraz kilka najbliższych słów
        print(embedding.closest("slow"))    # aby było widać 
        print(embedding.closest("car"))     # jak przebiega trenowanie modelu.
    
    
    
    # Najbliższe sobie słowa
    
    pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
             for w1 in vocab.w2i
             for w2 in vocab.w2i
             if w1 < w2]
    pairs.sort(reverse=True)
    print(pairs[:5])
    
    
    # Wyświetlenie wektorów słów
    plt.close()
    
    from scratch.working_with_data import pca, transform
    import matplotlib.pyplot as plt
    
    # Wyznacz pierwsze dwie główne składowe i przetransformuj wektory słów.
    components = pca(embedding.embeddings, 2)
    transformed = transform(embedding.embeddings, components)
    
    # Narysuj punkty na wykresie i pokoloruj je na biało, aby były niewidoczne.
    fig, ax = plt.subplots()
    ax.scatter(*zip(*transformed), marker='.', color='w')
    
    # Dodaj opis do każdego punktu.
    for word, idx in vocab.w2i.items():
        ax.annotate(word, transformed[idx])
    
    # Ukryj osie.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.show()
    
    
    
    plt.savefig('im/word_vectors')
    plt.gca().clear()
    plt.close()
    
    from bs4 import BeautifulSoup
    import requests
    
    url = "https://www.ycombinator.com/topcompanies/"
    soup = BeautifulSoup(requests.get(url).text, 'html5lib')
    
    # Pobieramy nazwy dwukrotnie, więc użyjemy zbioru, aby usunąć duplikaty.
    companies = list({b.text
                      for b in soup("b")
                      if "h4" in b.get("class", ())})
    assert len(companies) == 101
    
    vocab = Vocabulary([c for company in companies for c in company])
    
    START = "^"
    STOP = "$"
    
    # Należy je dodać do słownika.
    vocab.add(START)
    vocab.add(STOP)
    
    HIDDEN_DIM = 32  # Powinieneś poeksperymentować z różnymi rozmiarami!
    
    rnn1 =  SimpleRnn(input_dim=vocab.size, hidden_dim=HIDDEN_DIM)
    rnn2 =  SimpleRnn(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
    linear = Linear(input_dim=HIDDEN_DIM, output_dim=vocab.size)
    
    model = Sequential([
        rnn1,
        rnn2,
        linear
    ])
    
    from scratch.deep_learning import softmax
    
    def generate(seed: str = START, max_len: int = 50) -> str:
        rnn1.reset_hidden_state()  # Zresetuj obydwa ukryte stany
        rnn2.reset_hidden_state()
        output = [seed]            # rozpocznij od podstawienia pod output określonej wartości seed
    
        # Kontynuuj, aż trafisz na znak STOP lub do osiągnięcia maksymalnej długości 
        while output[-1] != STOP and len(output) < max_len:
            # Użyj ostatniego znaku na wejściu
            input = vocab.one_hot_encode(output[-1])
    
            # Wygeneruj wyniki, używając modelu
            predicted = model.forward(input)
    
            # Przekonwertuj je na prawdopodobieństwa i pobierz losowy char_id
            probabilities = softmax(predicted)
            next_char_id = sample_from(probabilities)
    
            # Dodaj odpowiedni znak do wartości wyjściowej
            output.append(vocab.get_word(next_char_id))
    
        # usuń znaki START i END i zwróć wygenerowane słowo
        return ''.join(output[1:-1])
    
    loss = SoftmaxCrossEntropy()
    optimizer = Momentum(learning_rate=0.01, momentum=0.9)
    
    for epoch in range(300):
        random.shuffle(companies)  # Za każdym przebiegiem zmieniamy kolejność.
        epoch_loss = 0             # Tśledzenie wartości straty.
        for company in tqdm.tqdm(companies):
            rnn1.reset_hidden_state()  # Zresetuj obydwa ukryte stany.
            rnn2.reset_hidden_state()
            company = START + company + STOP   # Add START and STOP characters.
    
            # Reszta działa jak typowa pętla treningowa, z tą różnicą, że wartości wejściowe oraz target są zakodowanym 
            # poprzednim i następnym znakiem.
            for prev, next in zip(company, company[1:]):
                input = vocab.one_hot_encode(prev)
                target = vocab.one_hot_encode(next)
                predicted = model.forward(input)
                epoch_loss += loss.loss(predicted, target)
                gradient = loss.gradient(predicted, target)
                model.backward(gradient)
                optimizer.step(model)
    
        # Przy każdym przebiegu wyświetl wartość straty oraz wygeneruj nazwę.
        print(epoch, epoch_loss, generate())
    
        # Zmniejszenie wartości learning rate na ostatnie 100 przebiegów.
        # Nie ma dobrze określonego powodu, by tak robić, ale wygląda na to, że działa to dobrze.
        if epoch == 200:
            optimizer.lr *= 0.1
    
if __name__ == "__main__": main()
