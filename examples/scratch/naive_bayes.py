from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Zamień na małe litery.
    all_words = re.findall("[a-z0-9']+", text)  # Wyciągnij słowa.
    return set(all_words)                       # Usuń duplikaty.

assert tokenize("Data Science is science") == {"data", "science", "is"}

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # czynnik wygładzający

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Zwiększ liczniki wiadomości
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Zwiększ liczniki słów
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Zwraca P(token|spam) oraz P(token|ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iteruj po wszystkich słowach w słowniku
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # Jeżeli w wiadomości pojawia się *token*,
            # dodaj logarytm prawdopodobieństwa napotkania takiego tokena 
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # W przeciwnym wypadku dodaj logarytm prawdopodobieństwa nienapotkania takiego tokena,
            # czyli log(1 - prawdopodobieństwo napotkania)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # słowo "spam" 
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # brak słowa "ham" 
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # brak słowa "rules" 
    (0 + 0.5) / (1 + 2 * 0.5)       # słowo "hello" 
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # słowo "spam" 
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # brak słowa "ham" 
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # brak słowa "rules" 
    (1 + 0.5) / (2 + 2 * 0.5),      # słowo "hello" 
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Powinno wyjść około 0.83
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

def drop_final_s(word):
    return re.sub("s$", "", word)

def main():
    import glob, re
    
    # ustaw ścieżkę do swojego folderu
    path = 'spam_data/*/*'
    
    data: List[Message] = []
    
    # glob.glob zwraca każdą nazwę pliku, która pasuje do podanej ścieżki
    for filename in glob.glob(path):
        is_spam = "ham" not in filename
    
        # W e-mailach znajduje się trochę nietypowych znaków. Opcja errors='ignore'
        # powoduje pomijanie ich zamiast wyrzucenia wyjątku.
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  # ten plik jest już gotowy
    
    import random
    from scratch.machine_learning import split_data
    
    random.seed(0)      # W celu ujednolicenia uzyskanych wyników.
    train_messages, test_messages = split_data(data, 0.75)
    
    model = NaiveBayesClassifier()
    model.train(train_messages)
    
    from collections import Counter
    
    predictions = [(message, model.predict(message.text))
                   for message in test_messages]
    
    # Załóż, że spam_probability > 0.5 oznacza to, że dana wiadomość jest spamem.
    # Policz kombinacje par (rzeczywista etykieta is_spam, przewidywana etykieta is_spam).
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)
    
    print(confusion_matrix)
    
    def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
        # Nie powinniśmy wywoływać prywatnej funkcji, ale tutaj robimy to w dobrej wierze.
        prob_if_spam, prob_if_ham = model._probabilities(token)
    
        return prob_if_spam / (prob_if_spam + prob_if_ham)
    
    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))
    
    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])
    
if __name__ == "__main__": main()
