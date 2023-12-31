# -*- coding: utf-8 -*-
from __future__ import division
import math, random, re, datetime
from collections import defaultdict, Counter
from functools import partial
from naive_bayes import tokenize

def word_count_old(documents):
    """Określanie liczby słów bez algorytmu MapReduce."""
    return Counter(word 
        for document in documents 
        for word in tokenize(document))

def wc_mapper(document):
    """Dla każdego słowa w dokumencie wygeneruj parę (słowo, 1)."""        
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word, counts):
    """Sumuj wystąpienia słowa"""
    yield (word, sum(counts))

def word_count(documents):
    """Policz słowa występujące w dokumencie wejściowym za pomocą algorytmu MapReduce"""

    # Tu będą przechowywane pogrupowane wartości.
    collector = defaultdict(list) 

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.iteritems()
            for output in wc_reducer(word, counts)]

def map_reduce(inputs, mapper, reducer):
    """Przetwarza elementy wejściowe za pomocą algorytmu MapReduce."""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.iteritems()
            for output in reducer(key,values)]

def reduce_with(aggregation_fn, key, values):
    """Redukuje pary klucz-wartość poprzez przetworzenie wartości za pomocą funkcji agregacji."""
    yield (key, aggregation_fn(values))

def values_reducer(aggregation_fn):
    """Zamienia funkcję (values -> output) na mechanizm redukujący."""
    return partial(reduce_with, aggregation_fn)

sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))

# 
# Analiza treści statusów
#

status_updates = [
    {"id": 1, 
     "username" : "joelgrus", 
     "text" : "Is anyone interested in a data science book?",
     "created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
     "liked_by" : ["data_guy", "data_gal", "bill"] },
    # add your own
]

def data_science_day_mapper(status_update):
    """Zwraca pary (dzień tygodnia, 1), jeżeli status zawiera frazę data science."""
    if "data science" in status_update["text"].lower():
        day_of_week = status_update["created_at"].weekday()
        yield (day_of_week, 1)
        
data_science_days = map_reduce(status_updates, 
                               data_science_day_mapper, 
                               sum_reducer)

def words_per_user_mapper(status_update):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
        yield (user, (word, 1))
            
def most_popular_word_reducer(user, words_and_counts):
    """Na podstawie sekwencji par (słowo, liczebność) 
    zwraca słowo pojawiające się najczęściej."""
    
    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count

    word, count = word_counts.most_common(1)[0]
                       
    yield (user, (word, count))

user_words = map_reduce(status_updates,
                        words_per_user_mapper, 
                        most_popular_word_reducer)

def liker_mapper(status_update):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)
                
distinct_likers_per_user = map_reduce(status_updates, 
                                      liker_mapper, 
                                      count_distinct_reducer)


#
# Mnożenie macierzy
#

def matrix_multiply_mapper(m, element):
    """Wspólnym wymiarem (kolumn macierzy A, wierszy macierzy B) jest m.
    Przetwarzane elementy mają formę krotki(nazwa_macierzy, i, j, wartość)."""
    matrix, i, j, value = element

    if matrix == "A":
        for column in range(m):
            # A_ij jest j-tym elementem sumy dla każdej kolumny C_i_column.
            yield((i, column), (j, value))
    else:
        for row in range(m):
            # B_ij jest i-tym elementem wiersza C_row_j.
            yield((row, j), (i, value))
     
def matrix_multiply_reducer(m, key, indexed_values):
    results_by_index = defaultdict(list)
    for index, value in indexed_values:
        results_by_index[index].append(value)

    # Sumuj wszystkie iloczyny pozycji.
    sum_product = sum(results[0] * results[1]
                      for results in results_by_index.values()
                      if len(results) == 2)
                      
    if sum_product != 0.0:
        yield (key, sum_product)

if __name__ == "__main__":

    documents = ["data science", "big data", "science fiction"]

    wc_mapper_results = [result 
                         for document in documents
                         for result in wc_mapper(document)]

    print "wc_mapper results"
    print wc_mapper_results
    print 

    print "word count results"
    print word_count(documents)
    print

    print "word count using map_reduce function"
    print map_reduce(documents, wc_mapper, wc_reducer)
    print

    print "data science days"
    print data_science_days
    print

    print "user words"
    print user_words
    print

    print "distinct likers"
    print distinct_likers_per_user
    print

    # matrix multiplication

    entries = [("A", 0, 0, 3), ("A", 0, 1,  2),
           ("B", 0, 0, 4), ("B", 0, 1, -1), ("B", 1, 0, 10)]
    mapper = partial(matrix_multiply_mapper, 3)
    reducer = partial(matrix_multiply_reducer, 3)

    print "map-reduce matrix multiplication"
    print "entries:", entries
    print "result:", map_reduce(entries, mapper, reducer)

    