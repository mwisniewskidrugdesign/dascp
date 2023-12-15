from typing import NamedTuple

class User(NamedTuple):
    id: int
    name: str

users = [User(0, "Hero"), User(1, "Dunn"), User(2, "Sue"), User(3, "Chi"),
         User(4, "Thor"), User(5, "Clive"), User(6, "Hicks"),
         User(7, "Devin"), User(8, "Kate"), User(9, "Klein")]

friend_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

from typing import Dict, List

# alias typu
Friendships = Dict[int, List[int]]

friendships: Friendships = {user.id: [] for user in users}

for i, j in friend_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

assert friendships[4] == [3, 5]
assert friendships[8] == [6, 7, 9]

from collections import deque

Path = List[int]

def shortest_paths_from(from_user_id: int,
                        friendships: Friendships) -> Dict[int, List[Path]]:
    # Słownik przypisujący najkrótsze ścieżki do identyfikatorów użytkowników.
    shortest_paths_to: Dict[int, List[Path]] = {from_user_id: [[]]}

    # Kolejka składa się z par (previous user, next user), które musimy sprawdzić.
    # Zaczyna się od wszystkich par (from_user, friend_of_from_user).
    frontier = deque((from_user_id, friend_id)
                     for friend_id in friendships[from_user_id])

    # Wykonuj tę operację aż do uzyskania pustej kolejki.
    while frontier:
        # Usuń następną parę z kolejki
        prev_user_id, user_id = frontier.popleft()

        # W związku ze sposobem dodawania użytkowników do kolejki
        # znamy już niektóre najkrótsze ścieżki prowadzące do użytkownika prev_user.
        paths_to_prev_user = shortest_paths_to[prev_user_id]
        new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]

        # Być może znamy już najkrótszą ścieżkę prowadzącą do tego użytkownika.
        old_paths_to_user = shortest_paths_to.get(user_id, [])

        # Jaka jest najkrótsza ścieżka prowadząca tutaj, której jeszcze nie widzieliśmy?
        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')

        # OZachowuj tylko nowe ścieżki, które nie są zbyt długie.
        new_paths_to_user = [path
                             for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]

        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

        # Dodaj niespotkanych wcześniej sąsiadów do listy frontier.
        frontier.extend((user_id, friend_id)
                        for friend_id in friendships[user_id]
                        if friend_id not in shortest_paths_to)

    return shortest_paths_to

# Lista najkrótszych ścieżek między każdym użytkownikiem from_user a każdym użytkownikiem to_user.
shortest_paths = {user.id: shortest_paths_from(user.id, friendships)
                  for user in users}

betweenness_centrality = {user.id: 0.0 for user in users}

for source in users:
    for target_id, paths in shortest_paths[source.id].items():
        if source.id < target_id:      # Nie sumuj dwukrotnie.
            num_paths = len(paths)     # Ile jest najkrótszych ścieżek?
            contrib = 1 / num_paths    # Określ wpływ na stopień pośrednictwa.
            for path in paths:
                for between_id in path:
                    if between_id not in [source.id, target_id]:
                        betweenness_centrality[between_id] += contrib

def farness(user_id: int) -> float:
    """Suma długości ścieżek wiodących do każdego użytkownika"""
    return sum(len(paths[0])
               for paths in shortest_paths[user_id].values())

closeness_centrality = {user.id: 1 / farness(user.id) for user in users}

from scratch.linear_algebra import Matrix, make_matrix, shape

def matrix_times_matrix(m1: Matrix, m2: Matrix) -> Matrix:
    nr1, nc1 = shape(m1)
    nr2, nc2 = shape(m2)
    assert nc1 == nr2, "liczba kolumn m1 musi być równa liczbie wierszy m2"

    def entry_fn(i: int, j: int) -> float:
        """Wynik funkcji dot na i-tym rzędzie macierzy m1 i j-tej kolumnie macierzy m2"""
        return sum(m1[i][k] * m2[k][j] for k in range(nc1))

    return make_matrix(nr1, nc2, entry_fn)

from scratch.linear_algebra import Vector, dot

def matrix_times_vector(m: Matrix, v: Vector) -> Vector:
    nr, nc = shape(m)
    n = len(v)
    assert nc == n, "liczba kolumn w m musi być równa liczbie elementów v"

    return [dot(row, v) for row in m]  # wynik ma długość nr

from typing import Tuple
import random
from scratch.linear_algebra import magnitude, distance

def find_eigenvector(m: Matrix,
                     tolerance: float = 0.00001) -> Tuple[Vector, float]:
    guess = [random.random() for _ in m]

    while True:
        result = matrix_times_vector(m, guess)    # przekształcenie wartości losowej
        norm = magnitude(result)                  # wyliczenie współczynnika normalizującego
        next_guess = [x / norm for x in result]   # przeskalowanie

        if distance(guess, next_guess) < tolerance:
            # punkt zbieżności, więc zwracamy (wektor własny, wartość własna)
            return next_guess, norm

        guess = next_guess

rotate = [[ 0, 1],
          [-1, 0]]

flip = [[0, 1],
        [1, 0]]

def entry_fn(i: int, j: int):
    return 1 if (i, j) in friend_pairs or (j, i) in friend_pairs else 0

n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
                (2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
                (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

from collections import Counter

endorsement_counts = Counter(target for source, target in endorsements)

import tqdm

def page_rank(users: List[User],
              endorsements: List[Tuple[int, int]],
              damping: float = 0.85,
              num_iters: int = 100) -> Dict[int, float]:
    outgoing_counts = Counter(target for source, target in endorsements)

    # Na początku wartości PageRank są rozdzielane po równo.
    num_users = len(users)
    pr = {user.id : 1 / num_users for user in users}

    # Podczas każdej iteracji każdy węzeł dostaje
    # niewielki ułamek wartości PageRank.
    base_pr = (1 - damping) / num_users

    for iter in tqdm.trange(num_iters):
        next_pr = {user.id : base_pr for user in users}  # zacznij od base_pr

        for source, target in endorsements:
            # Dodaj część pr z węzła źródłowego do węzłów docelowych
            next_pr[target] += damping * pr[source] / outgoing_counts[source]

        pr = next_pr

    return pr

pr = page_rank(users, endorsements)

# Thor (user_id 4) ma wyższy page rank od innych użytkowników
assert pr[4] > max(page_rank
                   for user_id, page_rank in pr.items()
                   if user_id != 4)

