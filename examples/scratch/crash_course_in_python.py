
"""
To są kody wprowadzające do Pythona.
Nie będą używane w dalszej części książki.
"""
# type: ignore

# znak # oznacza początek komentarza. Python ignoruje komentarze,
# ale mogą one być pomocne podczas czytania kodu
for i in [1, 2, 3, 4, 5]:
    print(i)                    # pierwsza linia bloku „for i”
    for j in [1, 2, 3, 4, 5]:
        print(j)                # pierwsza linia bloku „for j”
        print(i + j)            # ostatnia linia bloku „for j”
    print(i)                    # ostatnia linia bloku „for i”
print("koniec pętli")

dluga_operacja_arytmetyczna = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 +
                           13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)

lista_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

bardziej_czytelna_lista_list = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]

two_plus_three = 2 + \
                 3

for i in [1, 2, 3, 4, 5]:

    # zwróć uwagę na pustą linię
    print(i)

import re
my_regex = re.compile("[0-9]+", re.I)

import re as regex
my_regex = regex.compile("[0-9]+", regex.I)

from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

match = 10
from re import *    # O nie, moduł re zawiera funkcję o nazwie match.
print(match)        # "<function match at 0x10281e6a8>"

def double(x):
    """
    Tu możesz wstawić wyjaśnienie działania funkcji.
    Ta funkcja mnoży przekazaną do niej wartość przez 2
    """
    return x * 2

def apply_to_one(f):
    """Wywołuje funkcję f z argumentem 1"""
    return f(1)

my_double = double             # odwołanie do zdefiniowanej wcześniej funkcji
x = apply_to_one(my_double)    # równa się 2


assert x == 2

y = apply_to_one(lambda x: x + 4)      # równa się 5


assert y == 5

another_double = lambda x: 2 * x       # Nie rób tego.

def another_double(x):
    """Rób to tak."""
    return 2 * x

def my_print(message = "domyślny komunikat"):
    print(message)

my_print("Witaj")   # Wyświetli łańcuch „Witaj”.
my_print()          # Wyświetli łańcuch „domyślny komunikat”.

def full_name(first = "Jak-mu-tam", last = "Jakiś"):
    return first + " " + last

full_name("Joel", "Grus")     # zwraca "Joel Grus"
full_name("Joel")             # zwraca "Joel Jakiś"
full_name(last="Grus")        # zwraca "Jak-mu-tam Grus"


assert full_name("Joel", "Grus")     == "Joel Grus"
assert full_name("Joel")             == "Joel Jakiś"
assert full_name(last="Grus")        == "Jak-mu-tam Grus"

single_quoted_string = 'analiza danych'
double_quoted_string = "analiza danych"

tab_string = "\t"       # symbol znaku tabulacji
len(tab_string)         # długość łańcucha jest równa 1


assert len(tab_string) == 1

not_tab_string = r"\t"  # łańcuch zawierający znak ukośnika i literę t
len(not_tab_string)     # długość łańcucha jest równa 2


assert len(not_tab_string) == 2

multi_line_string = """To pierwsza linia,
to druga linia,
a to trzecia linia."""

first_name = "Joel"
last_name = "Grus"

full_name1 = first_name + " " + last_name             # dodawanie łańcuchów
full_name2 = "{0} {1}".format(first_name, last_name)  # string.format

full_name3 = f"{first_name} {last_name}"

try:
    print(0 / 0)
except ZeroDivisionError:
    print("nie można dzielić przez zero")

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)     # równa się 3
list_sum    = sum(integer_list)     # równa się 6


assert list_length == 3
assert list_sum == 6

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

zero = x[0]          # równa się 0, listy są indeksowane od zera
one = x[1]           # równa się 1
nine = x[-1]         # równa się 9, pythonowy sposób uzyskiwania ostatniego elementu
eight = x[-2]        # równa się 8, pythonowy sposób uzyskiwania przedostatniego elementu
x[0] = -1            # teraz x jest listą [-1, 1, 2, 3, ..., 9]


assert x == [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

first_three = x[:3]                 # [-1, 1, 2]
three_to_end = x[3:]                # [3, 4, ..., 9]
one_to_four = x[1:5]                # [1, 2, 3, 4]
last_three = x[-3:]                 # [7, 8, 9]
without_first_and_last = x[1:-1]    # [1, 2, ..., 8]
copy_of_x = x[:]                    # [-1, 1, 2, ..., 9]

every_third = x[::3]                 # [-1, 3, 6, 9]
five_to_three = x[5:2:-1]            # [5, 4, 3]


assert every_third == [-1, 3, 6, 9]
assert five_to_three == [5, 4, 3]

1 in [1, 2, 3]    # prawda (True)
0 in [1, 2, 3]    # fałsz (False)

x = [1, 2, 3]
x.extend([4, 5, 6])     # lista x ma teraz postać [1,2,3,4,5,6]


assert x == [1, 2, 3, 4, 5, 6]

x = [1, 2, 3]
y = x + [4, 5, 6]       # lista y ma teraz postać [1, 2, 3, 4, 5, 6]; lista x nie jest modyfikowana


assert x == [1, 2, 3]
assert y == [1, 2, 3, 4, 5, 6]

x = [1, 2, 3]
x.append(0)      # lista x ma teraz postać [1, 2, 3, 0]
y = x[-1]        # równa się 0
z = len(x)       # równa się 4


assert x == [1, 2, 3, 0]
assert y == 0
assert z == 4

x, y = [1, 2]    # teraz x jest równe 1, a y jest równe 2


assert x == 1
assert y == 2

_, y = [1, 2]    # teraz y == 2, a pierwszy element listy został pominięty

moja_lista = [1, 2]
moja_krotka = (1, 2)
inna_krotka = 3, 4
my_list[1] = 3      # moja_lista ma teraz postać [1, 3]

try:
    moja_krotka[1] = 3
except TypeError:
    print("nie można modyfikować krotki")

def sum_and_product(x, y):
    return (x + y), (x * y)

sp = sum_and_product(2, 3)     # równa się (5, 6)
s, p = sum_and_product(5, 10)  # s jest równe 15, p jest równe 50

x, y = 1, 2     # Teraz x jest równe 1, a y jest równe 2.
x, y = y, x     # Pythonowy sposób zamiany wartości zmiennych; teraz x jest równe 2, a y jest równe 1.


assert x == 2
assert y == 1

empty_dict = {}                     # zapis pythonowy
empty_dict2 = dict()                # zapis mniej pythonowy
grades = {"Joel": 80, "Tim": 95}    # literał słownikowy

joels_grade = grades["Joel"]        # równa się 80


assert joels_grade == 80

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("Kate nie ma żadnych ocen!")

joel_has_grade = "Joel" in grades     # prawda
kate_has_grade = "Kate" in grades     # fałsz


assert joel_has_grade
assert not kate_has_grade

joels_grade = grades.get("Joel", 0)   # równa się 80
kates_grade = grades.get("Kate", 0)   # równa się 0
no_ones_grade = grades.get("No One")  # domyślna wartość to None


assert joels_grade == 80
assert kates_grade == 0
assert no_ones_grade is None

grades["Tim"] = 99                    # zastępuje poprzednią wartość
grades["Kate"] = 100                  # dodaje trzeci element
num_students = len(grades)            # równa się 3


assert num_students == 3

tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

tweet_keys   = tweet.keys()     # lista kluczy
tweet_values = tweet.values()   # lista wartości
tweet_items  = tweet.items()    # lista krotek mających postać (klucz, wartość)

"user" in tweet_keys            # prawda, ale niezbyt pythonowy sposób
"user" in tweet                 # pythonowy sposób sprawdzania kluczy
"joelgrus" in tweet_values      # prawda (powolny, ale jedyny sposób, aby to sprawdzić)


assert "user" in tweet_keys
assert "user" in tweet
assert "joelgrus" in tweet_values


document = ["data", "science", "from", "scratch"]

word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

from collections import defaultdict

word_counts = defaultdict(int)          # int() generuje wartość 0
for word in document:
    word_counts[word] += 1

dd_list = defaultdict(list)             # Funkcja list() generuje pustą listę.
dd_list[2].append(1)                    # Teraz dd_list zawiera {2: [1]}.

dd_dict = defaultdict(dict)             # Funkcja dict() generuje pusty słownik.
dd_dict["Joel"]["City"] = "Seattle"     # { "Joel" : { "City" : Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1                       # Teraz dd_pair zawiera {2: [0,1]}.

from collections import Counter
c = Counter([0, 1, 2, 0])          # Obiekt c ma formę: { 0 : 2, 1 : 1, 2 : 1 }.

# document jest listą słów
word_counts = Counter(document)

# Wyświetl 10 najczęściej występujących słów i podaj liczbę ich wystąpień.
for word, count in word_counts.most_common(10):
    print(word, count)

primes_below_10 = {2, 3, 5, 7}

s = set()
s.add(1)       # s zawiera teraz {1}
s.add(2)       # s zawiera teraz {1, 2}
s.add(2)       # s wciąż zawiera {1, 2}
x = len(s)     # równa się 2
y = 2 in s     # zwraca prawdę
z = 3 in s     # zwraca fałsz


setki_innych_wyrazow = []  # required for the below code to run

stopwords_list = ["a", "an", "at"] + setki_innych_wyrazow + ["yet", "you"]

"zip" in stopwords_list     # Zwracany jest fałsz, ale operacja wymagała sprawdzenia każdego elementu.

stopwords_set = set(stopwords_list)
"zip" in stopwords_set      # Operacja sprawdzania przebiega bardzo szybko.

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)                # 6
item_set = set(item_list)                 # {1, 2, 3}
num_distinct_items = len(item_set)        # 3
distinct_item_list = list(item_set)       # [1, 2, 3]


assert num_items == 6
assert item_set == {1, 2, 3}
assert num_distinct_items == 3
assert distinct_item_list == [1, 2, 3]

if 1 > 2:
    message = "Gdyby 1 było większe od 2..."
elif 1 > 3:
    message = "Instrukcja elif służy do podawania kolejnego sprawdzanego warunku."
else:
    message = "Instrukcja else służy do definiowania kodu wykonywanego po niespełnieniu wszystkich warunków."

parity = "parzyste" if x % 2 == 0 else "nieparzyste"

x = 0
while x < 10:
    print(f"{x} jest mniejsze od 10")
    x += 1

# range(10) oznacza numery 0, 1, ..., 9
for x in range(10):
    print(f"{x} jest mniejsze od 10")

for x in range(10):
    if x == 3:
        continue  # Przejdź od razu do kolejnej iteracji.
    if x == 5:
        break     # Przerwij działanie pętli.
    print(x)

one_is_less_than_two = 1 < 2          # prawda logiczna (True)
true_equals_false = True == False     # fałsz logiczny (False)


assert one_is_less_than_two
assert not true_equals_false

x = None
assert x == None, "to nie jest pythonowy sposób na sprawdzenie, czy zmienna jest równa None"
assert x is None, "to jest pythonowy sposób na sprawdzenie, czy zmienna jest równa None"


def some_function_that_returns_a_string():
    return ""

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""

first_char = s and s[0]

safe_x = x or 0

safe_x = x if x is not None else 0

all([True, 1, {3}])   # True, wszystkie wartości są traktowane jako prawda.
all([True, 1, {}])    # False, {} jest traktowane jako fałsz.
any([True, 1, {}])    # True, True jest traktowane jako prawda.
all([])               # True, brak elementów będących fałszem.
any([])               # False, brak elementów będących prawdą.

x = [4, 1, 2, 3]
y = sorted(x)     # Lista y ma postać [1,2,3,4], a lista x pozostała niezmodyfikowana.
x.sort()          # Teraz lista x ma postać [1,2,3,4].

# Sortuje wartości od najwyższej do najniższej.
x = sorted([-4, 1, -2, 3], key=abs, reverse=True)  # is [-4, 3, -2, 1]

# Sortuje słowa i przypisane im wartości od najwyższej wartości do najniższej.
wc = sorted(word_counts.items(),
            key=lambda word_and_count: word_and_count[1],
            reverse=True)

even_numbers = [x for x in range(5) if x % 2 == 0]  # [0, 2, 4]
squares      = [x * x for x in range(5)]            # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]        # [0, 4, 16]


assert even_numbers == [0, 2, 4]
assert squares == [0, 1, 4, 9, 16]
assert even_squares == [0, 4, 16]

square_dict = {x: x * x for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
square_set  = {x * x for x in [1, -1]}      # {1}


assert square_dict == {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
assert square_set == {1}

zeros = [0 for _ in even_numbers]      # Obiekt wyjściowy ma taką samą długość jak even_numbers.


assert zeros == [0, 0, 0]

pairs = [(x, y)
         for x in range(10)
         for y in range(10)]   # 100 par (0,0) (0,1) ... (9,8), (9,9)


assert len(pairs) == 100

increasing_pairs = [(x, y)                       # tylko pary spełniające warunek x < y,
                    for x in range(10)           # range(pocz., koń.) jest równe
                    for y in range(x + 1, 10)]   # [pocz., pocz. + 1, ..., koń. - 1]


assert len(increasing_pairs) == 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
assert all(x < y for x, y in increasing_pairs)

assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 powinno być równe 2, ale nie jest"

def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

def smallest_item(xs):
    assert xs, "pusta lista nie ma najmniejszego elementu"
    return min(xs)

class CountingClicker:
    """Klasa, podobnie jak funkcja, powinna mieć opis dokumentujący. """

    def __init__(self, count = 0):
        self.count = count

    def __repr__(self):
        return f"CountingClicker(count={self.count})"

    def click(self, num_times = 1):
        """Kliknięcie licznika określoną liczbę razy."""
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0

clicker = CountingClicker()
assert clicker.read() == 0, "licznik powinien zaczynać od 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "po dwóch zliczeniach wartość licznika powinna wynosić 2"
clicker.reset()
assert clicker.read() == 0, "po zresetowaniu wartość licznika powinna wynosić 0"

# Klasa pochodna dziedziczy wszystkie cechy klasy bazowej.
class NoResetClicker(CountingClicker):
    # TTa klasa ma takie same funkcje jak CountingClicker

    # Z wyjątkiem tego, że funkcja reset nic nie robi.
    def reset(self):
        pass

clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "funkcja reset nie powinna niczego zmienić"

def generate_range(n):
    i = 0
    while i < n:
        yield i   # Każde odwołanie do funkcji yield powoduje wygenerowanie wartości generatora
        i += 1

for i in generate_range(10):
    print(f"i: {i}")

def natural_numbers():
    """Zwraca 1, 2, 3, ..."""
    n = 1
    while True:
        yield n
        n += 1

evens_below_20 = (i for i in generate_range(20) if i % 2 == 0)

# żadne z tych obliczeń nie będzie wykonane, dopóki nie przeprowadzimy iteracji
data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)
# i tak dalej


assert next(even_squares_ending_in_six) == 16
assert next(even_squares_ending_in_six) == 36
assert next(even_squares_ending_in_six) == 196

names = ["Alice", "Bob", "Charlie", "Debbie"]

# sposób niepythonowy
for i in range(len(names)):
    print(f"name {i} is {names[i]}")

# również niepythonowy sposób
i = 0
for name in names:
    print(f"name {i} is {names[i]}")
    i += 1

# sposób pythonowy
for i, name in enumerate(names):
    print(f"name {i} is {name}")

import random
random.seed(10)  # dzięki temu za każdym razem otrzymamy takie same rezultaty

four_uniform_randoms = [random.random() for _ in range(4)]

# [0.8444218515250481,       # Funkcja random.random() generuje liczby
#  0.7579544029403025,       # losowe rozłożone pomiędzy wartościami 0 i 1.
#  0.420571580830845,        # Jest to najczęściej używany przeze mnie sposób
#  0.25891675029296335]      # generowania liczb losowych.

random.seed(10)         # Ziarno przyjmuje wartość 10.
print(random.random())  # 0.57140259469
random.seed(10)         # Ponowne zdefiniowanie ziarna równego 10.
print(random.random())  # Wartość 0.57140259469 została wygenerowana ponownie.

random.randrange(10)    # Wylosuj liczbę z zakresu range(10) = [0, 1, ..., 9]
random.randrange(3, 6)  # Wylosuj liczbę z zakresu range(3, 6) = [3, 4, 5]

up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)
# [7, 2, 6, 8, 9, 4, 10, 1, 3, 5]   (prawdopodobnie uzyskasz inną kolejność)

my_best_friend = random.choice(["Alice", "Bob", "Charlie"])     # Wylosowałem imię Bob.

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)  # [16, 36, 10, 6, 25, 9]

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)  # [9, 4, 4, 2]

import re

re_examples = [                        # Wszystkie przykłady generują prawdę logiczną, ponieważ:
    not re.match("a", "cat"),              #  wyraz cat nie rozpoczyna się od litery a;
    re.search("a", "cat"),                 #  wyraz cat zawiera literę a;
    not re.search("c", "dog"),             #  wyraz dog nie zawiera litery c;
    3 == len(re.split("[ab]", "carbs")),   #  wyraz carbs po podzieleniu na literach a lub b daje listę ['c','r','s'];
    "R-D-" == re.sub("[0-9]", "-", "R2D2") #  cyfry zostają zastąpione kreskami.
    ]

assert all(re_examples), "wszystkie przykłady powinny zwracać True"

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# zip generuje wynik tylko wtedy, gdy jest potrzebny (jest tzw. funkcją leniwą). Można więc użyć jej np. tak:
[pair for pair in zip(list1, list2)]    # Generowana jest lista [('a', 1), ('b', 2), ('c', 3)].


assert [pair for pair in zip(list1, list2)] == [('a', 1), ('b', 2), ('c', 3)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

letters, numbers = zip(('a', 1), ('b', 2), ('c', 3))

def add(a, b): return a + b

add(1, 2)      # zwraca  3
try:
    add([1, 2])
except TypeError:
    print("funkcja add wymaga dwóch argumentów")
add(*[1, 2])   # zwraca  3

def doubler(f):
    # tutaj definiujemy nową funkcję, która korzysta z funkcji f
    def g(x):
        return 2 * f(x)

    # a tutaj zwracamy tę funkcję
    return g

def f1(x):
    return x + 1

g = doubler(f1)
assert g(3) == 8,  "(3 + 1) * 2 powinno być równe 8"
assert g(-1) == 0, "(-1 + 1) * 2 powinno być równe 0"

def f2(x, y):
    return x + y

g = doubler(f2)
try:
    g(1, 2)
except TypeError:
    print("funkcja g przyjmuje tylko jeden argument")

def magic(*args, **kwargs):
    print("argumenty nienazwane:", args)
    print("argumenty nazwane:", kwargs)

magic(1, 2, key="word", key2="word2")

# Kod wyświetli:
# argumenty nienazwane: (1, 2)
# argumenty nazwane: {'key2': 'word2', 'key': 'word'}

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 powinno być równe 6"

def doubler_correct(f):
    """Działa niezależnie od argumentów oczekiwanych przez funkcję f"""
    def g(*args, **kwargs):
        """Wszystkie argumenty funkcji g przekaż do funkcji f"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler powinien teraz działać poprawnie"

def add(a, b):
    return a + b

assert add(10, 5) == 15,                  "+ jest poprawne dla liczb"
assert add([1, 2], [3]) == [1, 2, 3],     "+ jest poprawne dla list"
assert add("hi ", "there") == "hi there", "+ jest poprawne dla łańcuchów"

try:
    add(10, "pięć")
except TypeError:
    print("nie można dodać liczby do łańcucha znakowego")

def add(a: int, b: int) -> int:
    return a + b

add(10, 5)           # to by było poprawne
add("hi ", "there")  # a to już nie


# Tego fragmentu nie ma w książce, ale jest potrzebny,
# aby funkcja `dot_product` nie zwracała błędu.
from typing import List
Vector = List[float]

def dot_product(x, y): ...

# jeszcze nie zdefiniowaliśmy typu Vector, ale wyobraź sobie, że to zrobiliśmy
def dot_product(x: Vector, y: Vector) -> float: ...

from typing import Union

def secretly_ugly_function(value, operation): ...

def ugly_function(value: int, operation: Union[str, int, float, bool]) -> int:
    ...

def total(xs: list) -> float:
    return sum(xs)

from typing import List  # pisane wielką literą L

def total(xs: List[float]) -> float:
    return sum(xs)

# tak wygląda adnotacja typu zmiennej podczas jej definiowania
# nie jest to konieczne, gdyż w tym przypadku jest oczywiste, że x jest typu int
x: int = 5

values = []         # jaki to typ?
best_so_far = None  # jaki to typ?

from typing import Optional

values: List[int] = []
best_so_far: Optional[float] = None  # zmienna może być albo typu float, albo None


lazy = True

# żadna z adnotacji typu w tym fragmencie nie jest konieczna
from typing import Dict, Iterable, Tuple

# klucze są typu string, a wartości typu int
counts: Dict[str, int] = {'data': 1, 'science': 2}

# # zarówno listy, jak i generatory są iterowalne
if lazy:
    evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
else:
    evens = [0, 2, 4, 6, 8]

# krotka z określonym typem każdego elementu
triple: Tuple[int, float, int] = (10, 2.3, 5)

from typing import Callable

# funkcja repeater przyjmuje dwa argumenty, jeden typu string, 
# a drugi typu int, i zwraca wartość typu string
def twice(repeater: Callable[[str, int], str], s: str) -> str:
    return repeater(s, 2)

def comma_repeater(s: str, n: int) -> str:
    n_copies = [s for _ in range(n)]
    return ', '.join(n_copies)

assert twice(comma_repeater, "adnotacja typu") == "adnotacja typu, adnotacja typu"

Number = int
Numbers = List[Number]

def total(xs: Numbers) -> Number:
    return sum(xs)
