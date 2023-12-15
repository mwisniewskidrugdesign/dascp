# -*- coding: utf-8 -*-
from __future__ import division
import math, random, re
from collections import defaultdict, Counter, deque
from linear_algebra import dot, get_row, get_column, make_matrix, magnitude, scalar_multiply, shape, distance
from functools import partial

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# give each user a friends list
for user in users:
    user["friends"] = []
    
# and populate it
for i, j in friendships:
    # Kod ten działa ponieważ users[i] odwołuje się do użytkownika o identyfikatorze i.
    users[i]["friends"].append(users[j]) # Dodaj i do listy przyjaciół j.
    users[j]["friends"].append(users[i]) # Dodaj j do listy przyjaciół i.

# 
# Betweenness Centrality
#

def shortest_paths_from(from_user):
    
    # Słownik przypisujący najkrótsze ścieżki do identyfikatorów użytkowników.
    shortest_paths_to = { from_user["id"] : [[]] }

    # Kolejka składa się z par(previous user, next user), które musimy sprawdzić.
    # Zaczyna się od wszystkich par(from_user, friend_of_from_user).
    frontier = deque((from_user, friend)
                     for friend in from_user["friends"])

    # Wykonuj tę operacje aż do uzyskania pustej kolejki.
    while frontier: 

        prev_user, user = frontier.popleft() # Usuń pierwszego użytkownika
        user_id = user["id"]                 # z kolejki.

        # W związku ze sposobem dodawania użytkowników do kolejki
        # znamy już niektóre najkrótsze ścieżki prowadzące do użytkownika prev_user.
        paths_to_prev = shortest_paths_to[prev_user["id"]]
        paths_via_prev = [path + [user_id] for path in paths_to_prev]
        
        # Być może znamy już najkrótszą ścieżkę prowadzącą do tego użytkownika.
        old_paths_to_here = shortest_paths_to.get(user_id, [])
        
        # Jaka jest najkrótsza ścieżka prowadząca tutaj, której jeszcze nie widzieliśmy?
        if old_paths_to_here:
            min_path_length = len(old_paths_to_here[0])
        else:
            min_path_length = float('inf')
                
        # Zachowuj tylko nowe ścieżki, które nie są zbyt długie.
        new_paths_to_here = [path_via_prev
                             for path_via_prev in paths_via_prev
                             if len(path_via_prev) <= min_path_length
                             and path_via_prev not in old_paths_to_here]
        
        shortest_paths_to[user_id] = old_paths_to_here + new_paths_to_here
        
        # Dodaj niespotkanych wcześniej sąsiadów do listy frontier.
        frontier.extend((user, friend)
                        for friend in user["friends"]
                        if friend["id"] not in shortest_paths_to)

    return shortest_paths_to

for user in users:
    user["shortest_paths"] = shortest_paths_from(user)

for user in users:
    user["betweenness_centrality"] = 0.0

for source in users:
    source_id = source["id"]
    for target_id, paths in source["shortest_paths"].iteritems():
        if source_id < target_id:   # Nie sumuj dwukrotnie.
            num_paths = len(paths)  # Ile jest najkrótszych ścieżek?
            contrib = 1 / num_paths # Określ wpływ na stopień pośrednictwa.
            for path in paths:
                for id in path:
                    if id not in [source_id, target_id]:
                        users[id]["betweenness_centrality"] += contrib

#
# closeness centrality
#

def farness(user):
    """Suma długości ścieżek wiodących do każdego użytkownika"""
    return sum(len(paths[0]) 
               for paths in user["shortest_paths"].values())

for user in users:
    user["closeness_centrality"] = 1 / farness(user)


#
# matrix multiplication
#

def matrix_product_entry(A, B, i, j):
    return dot(get_row(A, i), get_column(B, j))

def matrix_multiply(A, B):
    n1, k1 = shape(A)
    n2, k2 = shape(B)
    if k1 != n2:
        raise ArithmeticError("niekompatybilne kształty!")
                
    return make_matrix(n1, k2, partial(matrix_product_entry, A, B))

def vector_as_matrix(v):
    """Zwraca wektor v (mający formę listy) jako macierz n x 1."""
    return [[v_i] for v_i in v]
    
def vector_from_matrix(v_as_matrix):
    """Zwraca macierz n x 1 w formie listy wartości."""
    return [row[0] for row in v_as_matrix]

def matrix_operate(A, v):
    v_as_matrix = vector_as_matrix(v)
    product = matrix_multiply(A, v_as_matrix)
    return vector_from_matrix(product)

def find_eigenvector(A, tolerance=0.00001):
    guess = [1 for __ in A]

    while True:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/length, result)
        
        if distance(guess, next_guess) < tolerance:
            return next_guess, length # wektor własny, wartość własna
        
        guess = next_guess

#
# Centralność wektorów własnych
#

def entry_fn(i, j):
    return 1 if (i, j) in friendships or (j, i) in friendships else 0

n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)

eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)

#
# Grafy skierowane
#

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 3),
                (2, 3), (3, 4), (5, 4), (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

for user in users:
    user["endorses"] = []       # Pierwsza lista śledzi osoby polecane.
    user["endorsed_by"] = []    # Druga lista śledzi osoby polecające.
    
for source_id, target_id in endorsements:
    users[source_id]["endorses"].append(users[target_id])
    users[target_id]["endorsed_by"].append(users[source_id])


endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
                      for user in users]

sorted(endorsements_by_id, 
       key=lambda (user_id, num_endorsements): num_endorsements,
       reverse=True)

def page_rank(users, damping = 0.85, num_iters = 100):
    
    # Na początku wartości PageRank są rozdzielane  po równo.
    num_users = len(users)
    pr = { user["id"] : 1 / num_users for user in users }

    # Podczas każdej iteracji każdy węzeł dostaje
    # niewielki ułamek wartości PageRank.
    base_pr = (1 - damping) / num_users
    
    for __ in range(num_iters):
        next_pr = { user["id"] : base_pr for user in users }
        for user in users:
            # Rozdziel wartość PageRank pomiędzy węzłami wychodzącymi.
            links_pr = pr[user["id"]] * damping
            for endorsee in user["endorses"]:
                next_pr[endorsee["id"]] += links_pr / len(user["endorses"])

        pr = next_pr
        
    return pr

if __name__ == "__main__":

    print "Betweenness Centrality"
    for user in users:
        print user["id"], user["betweenness_centrality"]
    print

    print "Closeness Centrality"
    for user in users:
        print user["id"], user["closeness_centrality"]
    print

    print "Eigenvector Centrality"
    for user_id, centrality in enumerate(eigenvector_centralities):
        print user_id, centrality
    print

    print "PageRank"
    for user_id, pr in page_rank(users).iteritems():
        print user_id, pr
