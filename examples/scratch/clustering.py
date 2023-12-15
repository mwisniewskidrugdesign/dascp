from scratch.linear_algebra import Vector

def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0

from typing import List
from scratch.linear_algebra import vector_mean

def cluster_means(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    # clusters[i] zawiera punkty, które są przypisane do i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # jeżeli klaster jest pusty, użyj losowego punktu
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]

import itertools
import random
import tqdm
from scratch.linear_algebra import squared_distance

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k                      # liczba grup
        self.means = None

    def classify(self, input: Vector) -> int:
        """Zwróć indeks najbliższego klastra."""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # rozpocznij od losowego dopasowania.
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Oblicz średnie i znajdź nowe przypisania.
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Jeżeli przypisania do grup nie uległy zmianie, zakończ pracę.
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # W przeciwnym wypadku zachowaj przypisania.
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")

from typing import NamedTuple, Union

class Leaf(NamedTuple):
    value: Vector

leaf1 = Leaf([10,  20])
leaf2 = Leaf([30, -15])

class Merged(NamedTuple):
    children: tuple
    order: int

merged = Merged((leaf1, leaf2), order=1)

Cluster = Union[Leaf, Merged]

def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]

assert get_values(merged) == [[10, 20], [30, -15]]

from typing import Callable
from scratch.linear_algebra import distance

def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable = min) -> float:
    """
    Oblicz wszystkie odległości pomiędzy parami punktów z obydwu klastrów
    i skieruj uzyskaną listę wyników do funkcji _distance_agg_.
    """
    return distance_agg([distance(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])

def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float('inf')  # nigdy nie był łączony.
    else:
        return cluster.order

from typing import Tuple

def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children

def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable = min) -> Cluster:
    # Zacznij od wszystkich liści.
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)

    # Jeżeli został nam więcej niż jeden klaster…
    while len(clusters) > 1:
        # Znajdź dwa najbliższe klastry.
        c1, c2 = min(((cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]),
                      key=pair_distance)

        # Usuń je z listy klastrów.
        clusters = [c for c in clusters if c != c1 and c != c2]

        # Połącz klastry, używając merge_order równego liczbie pozostałych klastrów
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # Dodaj informację o ich połączeniu
        clusters.append(merged_cluster)

    # Jeżeli pozostał już tylko jeden klaster, to zwróć go.
    return clusters[0]

def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> List[Cluster]:
    # Zacznij od listy zawierającej tylko klaster bazowy.
    clusters = [base_cluster]

    # Jeżeli mamy jeszcze zbyt mało klastrów…
    while len(clusters) < num_clusters:
        # Wybierz klaster połączony jako ostatni
        next_cluster = min(clusters, key=get_merge_order)
        # Usuń go z listy.
        clusters = [c for c in clusters if c != next_cluster]

        # Dodaj jego elementy składowe do listy (rozłącz je).
        clusters.extend(get_children(next_cluster))

    # Jeżeli mamy już wystarczającą liczbę klastrów…
    return clusters

def main():
    
    
    inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    
    random.seed(12)                   # Dzięki temu uzyskasz taki sam wynik jak ja.
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)   # sortowanie dla testów jednostkowych
    
    assert len(means) == 3
    
    # Sprawdź, czy średnie są takie, jakich oczekiwaliśmy
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
    
    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    
    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1
    
    from matplotlib import pyplot as plt
    
    def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
        """Określa sumę błędów podniesionych do kwadratu
        uzyskanych w wyniku działania algorytmu k średnich"""
        clusterer = KMeans(k)
        clusterer.train(inputs)
        means = clusterer.means
        assignments = [clusterer.classify(input) for input in inputs]
    
        return sum(squared_distance(input, means[cluster])
                   for input, cluster in zip(inputs, assignments))
    
    # Wykonaj wykres dla podziału od 1 grupy do len(inputs) grup.
    
    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("Suma kwadratow bledow")
    plt.title("Blad calkowity a liczba grup")
    plt.show()
    
    
    
    plt.savefig('im/total_error_vs_num_clusters')
    plt.gca().clear()
    
    image_path = r"girl_with_book.jpg"    # ścieżka pliku obrazu
    import matplotlib.image as mpimg
    img = mpimg.imread(image_path) / 256  # przeskalujmy, aby uzyskać wartości z przedziału od 0 do 1
    
    # .tolist() konwertuje tablicę NumPy na obiekt list
    pixels = [pixel.tolist() for row in img for pixel in row]
    
    clusterer = KMeans(5)
    clusterer.train(pixels)   # Operacja ta może być czasochłonna.
    
    def recolor(pixel: Vector) -> Vector:
        cluster = clusterer.classify(pixel)        # indeks najbliższej grupy
        return clusterer.means[cluster]            # średnia najbliższej grupy
    
    new_img = [[recolor(pixel) for pixel in row]   # Zmień kolor tego rzędu pikseli.
               for row in img]                     # Wykonaj tę operację dla każdego wiersza obrazu.
    
    
    plt.close()
    
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()
    
    
    
    plt.savefig('im/recolored_girl_with_book.jpg')
    plt.gca().clear()
    
    base_cluster = bottom_up_cluster(inputs)
    
    three_clusters = [get_values(cluster)
                      for cluster in generate_clusters(base_cluster, 3)]
    
    
    
    # posortuj od najmniejszego do największego
    tc = sorted(three_clusters, key=len)
    assert len(tc) == 3
    assert [len(c) for c in tc] == [2, 4, 14]
    assert sorted(tc[0]) == [[11, 15], [13, 13]]
    
    
    plt.close()
    
    for i, cluster, marker, color in zip([1, 2, 3],
                                         three_clusters,
                                         ['D','o','*'],
                                         ['r','g','b']):
        xs, ys = zip(*cluster)  # rozpakowywanie
        plt.scatter(xs, ys, color=color, marker=marker)
    
        # Wprowadź średnią klastra.
        x, y = vector_mean(cluster)
        plt.plot(x, y, marker='$' + str(i) + '$', color='black')
    
    plt.title("Miejsca zamieszkania (3 grupy, metoda bottom-up, minimum)")
    plt.xlabel("Liczba przecznic na wschod od centrum miasta ")
    plt.ylabel("Liczba przecznic na polnoc od centrum miasta ")
    plt.show()
    
    
    
    plt.savefig('im/bottom_up_clusters_min.png')
    plt.gca().clear()
    plt.close()
    
    
    
    base_cluster_max = bottom_up_cluster(inputs, max)
    three_clusters_max = [get_values(cluster)
                          for cluster in generate_clusters(base_cluster_max, 3)]
    
    for i, cluster, marker, color in zip([1, 2, 3],
                                         three_clusters_max,
                                         ['D','o','*'],
                                         ['r','g','b']):
        xs, ys = zip(*cluster)  # rozpakowywanie
        plt.scatter(xs, ys, color=color, marker=marker)
    
        # Wprowadź średnią klastra.
        x, y = vector_mean(cluster)
        plt.plot(x, y, marker='$' + str(i) + '$', color='black')
    
    plt.title("Miejsca zamieszkania (3 grupy, metoda bottom-up, maksimum)")
    plt.xlabel("Liczba przecznic na wschod od centrum miasta ")
    plt.ylabel("Liczba przecznic na polnoc od centrum miasta ")
    plt.savefig('im/bottom_up_clusters_max.png')
    plt.gca().clear()
    
if __name__ == "__main__": main()
