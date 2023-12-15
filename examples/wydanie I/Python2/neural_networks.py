# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
from functools import partial
from linear_algebra import dot
import math, random
import matplotlib
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """Perceptron zwraca wartość 1 lub 0."""
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))
    
def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """Przyjmuje sieć neuronową (w formie listy list list wag)
    i zwraca wartość wyjściową w wyniku zastosowania algorytmu propagacji jednokierunkowej."""

    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]             # Dodaj wartość progową.
        output = [neuron_output(neuron, input_with_bias) # Oblicz wartość wyjściową
                  for neuron in layer]                   # każdego neuronu
        outputs.append(output)                           # i zapamiętaj ją.

        # Wyjście tej warstwy jest wejściem kolejnej warstwy.
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target):

    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    # Wartość output * (1 - output) bierze się z pochodnej funkcji sigmoid.
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]
                     
    # Dostosuj wagi dla warstwy wyjściowej, przetwarzaj neurony pojedynczo.
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # Wsteczna propagacja błędów do warstwy ukrytej
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                      dot(output_deltas, [n[i] for n in network[-1]]) 
                     for i, hidden_output in enumerate(hidden_outputs)]

    # Dobierz wagi dla warstwy ukrytej, przetwarzaj neurony pojedynczo.
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def patch(x, y, hatch, color):
    """Zwróć obiekt „łaty” wykresu znajdujący się w określonym miejscu,
    o określonym kolorze i sposobie kreskowania."""
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color)


def show_weights(neuron_idx):
    weights = network[0][neuron_idx]
    abs_weights = map(abs, weights)

    grid = [abs_weights[row:(row+5)] # Zamień wagi na siatkę 5x5.
            for row in range(0,25,5)] # [weights[0:5], ..., weights[20:25]]

    ax = plt.gca() # Osie są potrzebne w celu wykonania kreskowania.

    ax.imshow(grid, # Uruchom funkcję plt.imshow.
              cmap=matplotlib.cm.binary, # Korzystaj z czarno-białej palety barw.
              interpolation='none') # Przedstaw wykres składający się z bloków.

    # Kreskowanie ujemnych wag.
    for i in range(5): # rząd
        for j in range(5): # kolumna
            if weights[5*i + j] < 0: # rząd i, kolumna j = wagi[5*i + j]
                # Dodaj czarne i białe kreski, aby były one widoczne na czarnym i białym tle.
                ax.add_patch(patch(j, i, '/', "white"))
                ax.add_patch(patch(j, i, '\\', "black"))
    plt.show()

if __name__ == "__main__":

    raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",
             
          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",
             
          """11111
             ....1
             11111
             1....
             11111""",
             
          """11111
             ....1
             11111
             ....1
             11111""",     
             
          """1...1
             1...1
             11111
             ....1
             ....1""",             
             
          """11111
             1....
             11111
             ....1
             11111""",   
             
          """11111
             1....
             11111
             1...1
             11111""",             

          """11111
             ....1
             ....1
             ....1
             ....1""",
             
          """11111
             1...1
             11111
             1...1
             11111""",    
             
          """11111
             1...1
             11111
             ....1
             11111"""]     

    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]
                
    inputs = map(make_digit, raw_digits)

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)   # W celu uzyskania powtarzalnych wyników.
    input_size = 25  # Każdy wektor wejściowy ma długość równą 25.
    num_hidden = 5   # Warstwa ukryta składa się z 5 neuronów.
    output_size = 10 # Potrzebujemy wektora wyjściowego o długości 10.

    # Każdy ukryty neuron ma jedną wagę na wejście oraz wagę wartości progowej.
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # Każdy neuron wyjściowy ma jedną wagę przypadającą na neuron ukryty oraz wagę wartości progowej.
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # Sieć na początku przyjmuje losowe wartości wag.
    network = [hidden_layer, output_layer]

    # Wydaje się, że 10 000 iteracji wystarczy do osiągnięcia punktu zbieżności.
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        return feed_forward(network, input)[-1]

    for i, input in enumerate(inputs):
        outputs = predict(input)
        print i, [round(p,2) for p in outputs]

    print """.@@@.
...@@
..@@.
...@@
.@@@."""
    print [round(x, 2) for x in
          predict(  [0,1,1,1,0,  # .@@@.
                     0,0,0,1,1,  # ...@@
                     0,0,1,1,0,  # ..@@.
                     0,0,0,1,1,  # ...@@
                     0,1,1,1,0]) # .@@@.
          ]
    print

    print """.@@@.
@..@@
.@@@.
@..@@
.@@@."""
    print [round(x, 2) for x in 
          predict(  [0,1,1,1,0,  # .@@@.
                     1,0,0,1,1,  # @..@@
                     0,1,1,1,0,  # .@@@.
                     1,0,0,1,1,  # @..@@
                     0,1,1,1,0]) # .@@@.
          ]
    print

    