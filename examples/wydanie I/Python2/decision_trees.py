# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter, defaultdict
from functools import partial
import math, random

def entropy(class_probabilities):
    """Oblicz entropię na podstawie listy prawdopodobieństw klas."""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):        
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """Określ entropię podziału danych na podzbiory;
	podzbiory mają formę listy list danych z etykietami."""
    total_count = sum(len(subset) for subset in subsets)
    
    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets )

def group_by(items, key_fn):
    """returns a defaultdict(list), where each input item 
    is in the list whose key is key_fn(item)"""
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups
    
def partition_by(inputs, attribute):
    """Dane wejściowe składają się ze słownika atrybutów i etykiety.
    Zwracany jest słownik: wartość atrybutu -> dane wejściowe"""
    return group_by(inputs, lambda x: x[0][attribute])    

def partition_entropy_by(inputs,attribute):
    """Oblicza entropię odpowiadającą danemu podziałowi."""        
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())        

def classify(tree, input):
    """Klasyfikuj dane wejściowe za pomocą danego drzewa decyzyjnego."""
    
    # Jeżeli jest to węzeł końcowy, to zwróć jego wartość.
    if tree in [True, False]:
        return tree
   
    # W przeciwnym wypadku drzewo to składa się z atrybutu, na bazie którego należy wykonać podział
    # i słownika, którego klucze są wartościami tego atrybutu.
    # Wartości słownika są poddrzewami, które należy rozpatrzyć w dalszej kolejności.

    attribute, subtree_dict = tree
    
    subtree_key = input.get(attribute)  # None pojawia się w przypadku brakującego atrybutu.

    if subtree_key not in subtree_dict: # Poddrzewo None zostanie użyte
        subtree_key = None              # w przypadku braku poddrzewa klucza.
    
    subtree = subtree_dict[subtree_key] # Wybierz właściwe poddrzewo
    return classify(subtree, input)     # i użyj go w celu przeprowadzenia klasyfikacji.

def build_tree_id3(inputs, split_candidates=None):

    # W przypadku pierwszej iteracji 
    # wszystkie klucze pierwszego elementu wejściowego są potencjalnymi kandydatami do podziału.
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # Policz wartości True i False występujące w danych wejściowych.
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    
    if num_trues == 0:                  # Zwróć węzeł False 
        return False                    # w przypadku braku wartości True.
        
    if num_falses == 0:                 # Zwróć węzeł True 
        return True                     # w przypadku braku wartości False.

    if not split_candidates:            # W przypadku braku kandydatów do podziału
        return num_trues >= num_falses  # zwróć węzeł określony przez większość wartości.
                            
    # W przeciwnym wypadku podziel na najlepszym atrybucie.
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates 
                      if a != best_attribute]
    
    # Buduj poddrzewa w sposób rekurencyjny.
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.iteritems() }

    subtrees[None] = num_trues > num_falses # przypadek domyślny

    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


if __name__ == "__main__":

    inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
    ]

    for key in ['level','lang','tweets','phd']:
        print key, partition_entropy_by(inputs, key)
    print

    senior_inputs = [(input, label)
                     for input, label in inputs if input["level"] == "Senior"]

    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(senior_inputs, key)
    print

    print "building the tree"
    tree = build_tree_id3(inputs)
    print tree

    print "Junior / Java / tweets / no phd", classify(tree, 
        { "level" : "Junior", 
          "lang" : "Java", 
          "tweets" : "yes", 
          "phd" : "no"} ) 

    print "Junior / Java / tweets / phd", classify(tree, 
        { "level" : "Junior", 
                 "lang" : "Java", 
                 "tweets" : "yes", 
                 "phd" : "yes"} )

    print "Intern", classify(tree, { "level" : "Intern" } )
    print "Senior", classify(tree, { "level" : "Senior" } )

