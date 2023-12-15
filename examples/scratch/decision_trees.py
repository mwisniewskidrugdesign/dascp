from typing import List
import math

def entropy(class_probabilities: List[float]) -> float:
    """Oblicz entropię na podstawie listy prawdopodobieństw klas."""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p > 0)                     # Ignoruj zerowe prawdopodobieństwa.

assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

from typing import Any
from collections import Counter

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])

def partition_entropy(subsets: List[List[Any]]) -> float:
    """Określ entropię podziału danych na podzbiory."""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)

from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # dopuszczaj dane bez etykiety

                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T')  # Generyczny typ dla wartości wejściowych

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Podziel dane wejściowe na listy na podstawie określonego atrybutu."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # Określ wartość wybranego atrybutu.
        partitions[key].append(input)    # Dodaj te dane wejściowe do właściwej listy.
    return partitions

def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """Oblicza entropię odpowiadającą danemu podziałowi."""
    # partitions składa się z danych wejściowych       
    partitions = partition_by(inputs, attribute)

    # ale partition_entropy potrzebuje jedynie etykiet klas       
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)

for key in ['level','lang','tweets','phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))

assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well')  < 0.70
assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well')   < 0.87
assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well')    < 0.90

senior_inputs = [input for input in inputs if input.level == 'Senior']

assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96

from typing import NamedTuple, Union, Any

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

hiring_tree = Split('level', {   # na początek rozdzielamy pod względem poziomu zawodowego
    'Junior': Split('phd', {     # jeżeli poziom to "Junior", dzielimy pod względem doktoratu
        False: Leaf(True),       # jeżeli brak doktoratu, zwraca True
        True: Leaf(False)        # jeżeli jest doktorat, zwraca False
    }),
    'Mid': Leaf(True),           # jeżeli poziom to "Mid", zwraca True
    'Senior': Split('tweets', {  # jeżeli poziom to "Senior", dzielimy pod względem aktywności na Twitterze
        False: Leaf(False),      # jeżeli nieaktywny na Twitterze, zwraca False
        True: Leaf(True)         # jeżeli aktywny na Twitterze, zwraca True
    })
})

def classify(tree: DecisionTree, input: Any) -> Any:
    """Klasyfikuj dane wejściowe za pomocą danego drzewa decyzyjnego."""

    # Jeżeli jest to węzeł końcowy, to zwróć jego wartość.
    if isinstance(tree, Leaf):
        return tree.value

    # W przeciwnym wypadku drzewo to składa się z atrybutu, na bazie którego należy wykonać podział,
    # i słownika, którego klucze są wartościami tego atrybutu.
    # Wartości słownika są poddrzewami, które należy rozpatrzyć w dalszej kolejności.
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:   # wartość domyślna zostanie użyta
        return tree.default_value          # w przypadku braku poddrzewa klucza.

    subtree = tree.subtrees[subtree_key]   # Wybierz właściwe poddrzewo
    return classify(subtree, input)        # i użyj go w celu przeprowadzenia klasyfikacji.

def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Policz etykiety 
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # jeżeli jest tylko jedna etykieta, zwróć ją 
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # jeżeli nie ma po czym dzielić, zwróć najbardziej liczną etykietę 
    if not split_attributes:
        return Leaf(most_common_label)

    # W przeciwnym wypadku podziel na najlepszym atrybucie.

    def split_entropy(attribute: str) -> float:
        """funkcja pomocnicza do znajdowania najlepszego atrybutu"""
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # Buduj poddrzewa w sposób rekurencyjny.
    subtrees = {attribute_value : build_tree_id3(subset,
                                                 new_attributes,
                                                 target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)

tree = build_tree_id3(inputs,
                      ['level', 'lang', 'tweets', 'phd'],
                      'did_well')

# powinien zwrócić True
assert classify(tree, Candidate("Junior", "Java", True, False))

# powinien zwrócić False
assert not classify(tree, Candidate("Junior", "Java", True, True))

# powinien zwrócić True
assert classify(tree, Candidate("Intern", "Java", True, True))

