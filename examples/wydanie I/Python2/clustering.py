# -*- coding: utf-8 -*-
from __future__ import division
from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KMeans:
    """Przeprowadza grupowanie metodą k średnich."""

    def __init__(self, k):
        self.k = k          # liczba grup
        self.means = None   # średnia klastrów
        
    def classify(self, input):
        """Zwróć indeks najbliższego klastra."""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))
                   
    def train(self, inputs):
    
        self.means = random.sample(inputs, self.k)
        assignments = None
        
        while True:
            # Znajdź nowe przypisania.
            new_assignments = map(self.classify, inputs)

            # Jeżeli przypisania do grup nie uległy zmianie, zakończ pracę.
            if assignments == new_assignments:                
                return

            # W przeciwnym wypadku zachowaj przypisania.
            assignments = new_assignments    

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # Upewnij się, że i_points nie jest pustym zbiorem, aby uniknąć dzielenia przez 0.
                if i_points:                                
                    self.means[i] = vector_mean(i_points)    

def squared_clustering_errors(inputs, k):
    """Określa sumę błędów podniesionych do kwadratu uzyskanych w wyniku działania algorytmu k średnich"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)
    
    return sum(squared_distance(input,means[cluster])
               for input, cluster in zip(inputs, assignments))

def plot_squared_clustering_errors(plt):

    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.show()

#
# using clustering to recolor an image
#

def recolor_image(input_file, k=5):

    img = mpimg.imread(path_to_png_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels) # Operacja ta może być czasochłonna.  

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # indeks najbliższej grupy
        return clusterer.means[cluster]     # średnia najbliższej grupy

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

#
# Grupowanie hierarchiczne
#

def is_leaf(cluster):
    """Klaster jest obiektem leaf jeżeli jego długość jest równa 1."""
    return len(cluster) == 1

def get_children(cluster):
    """Zwraca dwa elementy potomne klastra, jeżeli jest to klaster połączony;
    w przypadku klastra typu leaf zwrócony zostanie wyjątek."""
    if is_leaf(cluster):
        raise TypeError("Brak grup potomnych")
    else:
        return cluster[1]

def get_values(cluster):
    """Zwraca wartość klastra (jeżeli nie ma on elementów potomnych)
    lub wszystkie wartości jego klastrów pochodnych (w przeciwnym wypadku)."""
    if is_leaf(cluster):
        return cluster # Jest już jednoelementową krotką zawierającą wartość.
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]

def cluster_distance(cluster1, cluster2, distance_agg=min):
    """Oblicz wszystkie parzyste odległości pomiędzy klastrami
    i skieruj uzyskaną listę wyników do funkcji distance_agg."""
    return distance_agg([distance(input1, input2)
                        for input1 in get_values(cluster1)
                        for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0] # Wartość merge_order jest pierwszym elementem dwuelementowej krotki.

def bottom_up_cluster(inputs, distance_agg=min):
    # Zacznij od utworzenia jednoelementowych krotek.
    clusters = [(input,) for input in inputs]
    
    # Jeżeli został nam więcej niż jeden klaster...
    while len(clusters) > 1:
        # Znajdź dwa najbliższe klastry.
        c1, c2 = min([(cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]],
                     key=lambda (x, y): cluster_distance(x, y, distance_agg))

        # Usuń je z listy klastrów.
        clusters = [c for c in clusters if c != c1 and c != c2]

        # Połącz klastry.
        merged_cluster = (len(clusters), [c1, c2])

        # Dodaj informację o ich połączeniu.
        clusters.append(merged_cluster)

    # Jeżeli pozostał już tylko jeden klaster, to zwróć go.
    return clusters[0]

def generate_clusters(base_cluster, num_clusters):
    # Zacznij od listy zawierającej tylko klaster bazowy.
    clusters = [base_cluster]
    
    # Jeżeli mamy jeszcze zbyt mało klastrów...
    while len(clusters) < num_clusters:
        # Wybierz klaster połączony jako ostatni
        next_cluster = min(clusters, key=get_merge_order)
        # Usuń go z listy.
        clusters = [c for c in clusters if c != next_cluster]
        # Dodaj jego elementy składowe do listy (rozłącz je).
        clusters.extend(get_children(next_cluster))

    # Jeżeli mamy już wystarczającą liczbę klastrów...
    return clusters

if __name__ == "__main__":

    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(0) # Dzięki temu uzyskasz taki sam wynik jak ja.
    clusterer = KMeans(3)
    clusterer.train(inputs)
    print "3-means:"
    print clusterer.means
    print

    random.seed(0)
    clusterer = KMeans(2)
    clusterer.train(inputs)
    print "2-means:"
    print clusterer.means
    print

    print "errors as a function of k"

    for k in range(1, len(inputs) + 1):
        print k, squared_clustering_errors(inputs, k)
    print


    print "bottom up hierarchical clustering"

    base_cluster = bottom_up_cluster(inputs)
    print base_cluster

    print
    print "three clusters, min:"
    for cluster in generate_clusters(base_cluster, 3):
        print get_values(cluster)

    print
    print "three clusters, max:"
    base_cluster = bottom_up_cluster(inputs, max)
    for cluster in generate_clusters(base_cluster, 3):
        print get_values(cluster)
