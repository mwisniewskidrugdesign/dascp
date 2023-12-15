from scratch.linear_algebra import Vector, dot

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """Perceptron zwraca wartość 1 lub 0."""
    calculation = dot(weights, x) + bias
    return step_function(calculation)

and_weights = [2., 2]
and_bias = -3.

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

or_weights = [2., 2]
or_bias = -1.

assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0

not_weights = [-2.]
not_bias = 1.

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0

import math

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    # wektor weights ma na ostatniej pozycji wartość progową (bias), a wektor inputs ma na ostatniej pozycji wartość 1.
    return sigmoid(dot(weights, inputs))

from typing import List

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Przesyła wektor wejściowy przez sieć neuronową.
    Zwraca wyjścia wszystkich warstw (nie tylko ostatniej).
    """
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]              # Dodaj wartość stałą (input bias).
        output = [neuron_output(neuron, input_with_bias)  # Oblicz wartość wyjściową
                  for neuron in layer]                    # każdego neuronu
        outputs.append(output)                            # i zapamiętaj ją.

        # Wyjście tej warstwy jest wejściem kolejnej warstwy.
        input_vector = output

    return outputs

xor_network = [# warstwa ukryta
               [[20., 20, -30],      # neuron AND
                [20., 20, -10]],     # neuron OR
               # warstwa wyjściowa
               [[-60., 60, -30]]]    # drugie wejście AND NOT wejście pierwsze

# Funkcja feed_forward generuje wartości wyjściowe każdego neuronu.
# Funkcja feed_forward[-1] zwraca dane wyjściowe neuronów warstwy wyjściowej.
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Na wejściu dostaje sieć neuronową, wektor wejściowy i wektor wyjściowy.
    Przelicza sieć, a następnie oblicza gradient błędu kwadratowego w odniesieniu do wag neuronów. 
    """
    # przeliczenie sieci
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradienty wartości wyjściowych końcowego neuronu
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradienty z uwzględnieniem wag neuronu wyjściowego
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradienty wartości wyjściowych ukrytych neuronów
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradienty z uwzględnieniem wag ukrytych neuronów
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

[   # warstwa ukryta
    [[7, 7, -3],     # oblicza OR
     [5, 5, -8]],    # oblicza AND
    # warstwa wyjściowa
    [[11, -12, -5]]  # oblicza "pierwszy, ale nie drugi"
]

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

assert fizz_buzz_encode(2) == [1, 0, 0, 0]
assert fizz_buzz_encode(6) == [0, 1, 0, 0]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(30) == [0, 0, 0, 1]

def binary_encode(x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary

#                             1  2  4  8 16 32 64 128 256 512
assert binary_encode(0)   == [0, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(1)   == [1, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(10)  == [0, 1, 0, 1, 0, 0, 0, 0,  0,  0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0,  0,  0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1,  1,  1]

def argmax(xs: list) -> int:
    """Zwraca indeks najwyższej wartości"""
    return max(range(len(xs)), key=lambda i: xs[i])

assert argmax([0, -1]) == 0               # items[0] ma największą wartość
assert argmax([-1, 0]) == 1               # items[1] ma największą wartość
assert argmax([-1, 10, 5, 20, -3]) == 3   # items[3] ma największą wartość

def main():
    import random
    random.seed(0)
    
    # dane treningowe
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]
    
    # rozpocznij od losowych wag
    network = [ # warstwa ukryta: 2 wartości wejściowe -> 2 wartości wyjściowe
                [[random.random() for _ in range(2 + 1)],   # pierwszy ukryty neuron 
                 [random.random() for _ in range(2 + 1)]],  # drugi ukryty neuron
                # warstwa wyjściowa: 2 wartości wejściowe -> 1 wynik
                [[random.random() for _ in range(2 + 1)]]   # pierwszy neuron wyjściowy
              ]
    
    from scratch.gradient_descent import gradient_step
    import tqdm
    
    learning_rate = 1.0
    
    for epoch in tqdm.trange(20000, desc="neural net for xor"):
        for x, y in zip(xs, ys):
            gradients = sqerror_gradients(network, x, y)
    
            # Zrób krok w kierunku gradientu dla każdego neuronu, w każdej warstwie.
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)]
    
    # sprawdź, czy faktycznie implementuje bramkę XOR
    assert feed_forward(network, [0, 0])[-1][0] < 0.01
    assert feed_forward(network, [0, 1])[-1][0] > 0.99
    assert feed_forward(network, [1, 0])[-1][0] > 0.99
    assert feed_forward(network, [1, 1])[-1][0] < 0.01
    
    xs = [binary_encode(n) for n in range(101, 1024)]
    ys = [fizz_buzz_encode(n) for n in range(101, 1024)]
    
    NUM_HIDDEN = 25
    
    network = [
        # warstwa ukryta: 10 wejść -> NUM_HIDDEN wyjść
        [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],
    
        # warstwa wyjściowa: NUM_HIDDEN wejść -> 4 wyjść
        [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
    ]
    
    from scratch.linear_algebra import squared_distance
    
    learning_rate = 1.0
    
    with tqdm.trange(500) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(xs, ys):
                predicted = feed_forward(network, x)[-1]
                epoch_loss += squared_distance(predicted, y)
                gradients = sqerror_gradients(network, x, y)
    
                # Zrób krok w kierunku gradientu dla każdego neuronu w każdej warstwie
                network = [[gradient_step(neuron, grad, -learning_rate)
                            for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]
    
            t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")
    
    num_correct = 0
    
    for n in range(1, 101):
        x = binary_encode(n)
        predicted = argmax(feed_forward(network, x)[-1])
        actual = argmax(fizz_buzz_encode(n))
        labels = [str(n), "fizz", "buzz", "fizzbuzz"]
        print(n, labels[predicted], labels[actual])
    
        if predicted == actual:
            num_correct += 1
    
    print(num_correct, "/", 100)
    
if __name__ == "__main__": main()
