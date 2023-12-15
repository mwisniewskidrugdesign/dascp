Tensor = list

from typing import List

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]

def is_1d(tensor: Tensor) -> bool:
    """
    Jeżeli tensor[0] jest listą, to jest to tensor wyższego rzędu.
    W przeciwnym wypadku jest to jednowymiarowy tensor (czyli wektor).
    """
    return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])

def tensor_sum(tensor: Tensor) -> float:
    """Sumuje wszystkie elementy tensora"""
    if is_1d(tensor):
        return sum(tensor)  # zwykła lista typu float, używamy pythonowej funkcji sum
    else:
        return sum(tensor_sum(tensor_i)      # wywołujemy tensor_sum na każdym wierszu
                   for tensor_i in tensor)   # i sumujemy wyniki.

assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10

from typing import Callable

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Wywołuje funkcję f dla każdego elementu"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]

assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Wywołuje funkcję f na odpowiadających elementach tensorów t1 i t2"""
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]

import operator
assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]

from typing import Iterable, Tuple

class Layer:
    """
    Nasza sieć neuronowa będzie składała się z warstw (Layer), z których każda
    potrafi przeprowadzić pewne obliczenia na wartościach wejściowych w kierunku 
    "do przodu" (forward) i propagować gradient "do tyłu" (backward). 
    """
    def forward(self, input):
        """
        Zauważ brak typów. Nie zamierzamy określać, jakie typy wartości
        mogą być przyjmowane na wejściu i jakie będą zwracane.
        """
        raise NotImplementedError

    def backward(self, gradient):
        """
        Podobnie nie zamierzamy określać, jak powinien wyglądać gradient.
        To użytkownik będzie musiał się upewnić, że to, co robi, ma sens.
        """
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """
        Zwraca parametry warstwy. Domyślna implementacja nie zwraca niczego, więc jeżeli
        korzystasz z warstwy bez parametrów, nie musisz implementować tej metody.
        """
        return ()

    def grads(self) -> Iterable[Tensor]:
        """
        Zwraca gradienty w taki sam sposób jak params().
        """
        return ()

from scratch.neural_networks import sigmoid

class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        """
        Wywołuje funkcję sigmoid dla każdego wejściowego tensora
        i zapisuje wyniki, by użyć ich w propagacji wstecznej.
        """
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids,
                              gradient)

import random

from scratch.probability import inverse_normal_cdf

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")

from scratch.linear_algebra import dot

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
        """
        Warstwa, dla której liczbę neuronów określa parametr output_dim, a liczba wag to input_dim (oraz bias).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.w[o] jest wagą o-tego neuronu
        self.w = random_tensor(output_dim, input_dim, init=init)

        # self.b[o] to wartość progowa(bias) o-tego neuronu
        self.b = random_tensor(output_dim, init=init)

    def forward(self, input: Tensor) -> Tensor:
        # Zachowaj wartość wejściową do wykorzystania w propagacji wstecznej.
        self.input = input

        # Zwróć wektor wyników z wszystkich neuronów.
        return [dot(input, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        # Każdy b[o] jest dodawany do output[o], 
        # co oznacza, że gradient b jest taki sam jak gradient output.
        self.b_grad = gradient

        # Każdy input[i] jest mnożony przez wagę w[o][i] i dodawany do output[o].
        # Jego gradient wynosi więc input[i] * gradient[o]. 
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]

        # Każdy input[i] jest mnożony przez każdą wagę w[o][i] i dodawany do każdego output[o].
        # Jego gradient jest więc sumą w[o][i] * gradient[o]we wszystkich wartościach wyjściowych
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]

from typing import List

class Sequential(Layer):
    """
    Warstwa składająca się z sekwencji innych warstw.
    Użytkownik musi zadbać o to, by wyjście każdej warstwy
    pasowało do wejścia kolejnej. 
    """
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        """Przesyłanie wartości wejściowych przez warstwy w odpowiednim porządku."""
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        """Przeprowadza wsteczną propagację gradientu przez wszystkie warstwy."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        """Zwraca parametry z każdej warstwy."""
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """Zwraca gradient z każdej warstwy."""
        return (grad for layer in self.layers for grad in layer.grads())

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """Jak dobre są nasze przewidywania? (Im większa wartość, tym gorzej.)"""
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """W jaki sposób zmienia się strata wraz ze zmianą przewidywań?"""
        raise NotImplementedError

class SSE(Loss):
    """Funkcja straty, która oblicza sumę kwadratów błędów."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Oblicz tensor różnic podniesionych do kwadratu
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual)

        # Dodaj je do siebie
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual)


sse_loss = SSE()
assert sse_loss.loss([1, 2, 3], [10, 20, 30]) == 9 ** 2 + 18 ** 2 + 27 ** 2
assert sse_loss.gradient([1, 2, 3], [10, 20, 30]) == [-18, -36, -54]

class Optimizer:
    """
    Obiekt Optimizer aktualizuje wagi warstwy (w miejscu), korzystając z informacji,
    które posiada warstwa lub obiekt Optimizer.
    """
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Zaktualizuj param przez zmianę w kierunku gradientu
            param[:] = tensor_combine(
                lambda param, grad: param - grad * self.lr,
                param,
                grad)

tensor = [[1, 2], [3, 4]]

for row in tensor:
    row = [0, 0]
assert tensor == [[1, 2], [3, 4]], "przypisanie nie zmienia listy"

for row in tensor:
    row[:] = [0, 0]
assert tensor == [[0, 0], [0, 0]], "ale przypisanie do wycinka już tak"

class Momentum(Optimizer):
    def __init__(self,
                 learning_rate: float,
                 momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []  # średnia krocząca

    def step(self, layer: Layer) -> None:
        # Jeżeli nie mamy wcześniejszych aktualizacji, zacznijmy od wszystkich równych zero.
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates,
                                       layer.params(),
                                       layer.grads()):
            # skorzystajmy ze średniej kroczącej
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad)

            # następnie zróbmy krok w kierunku gradientu
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update)

import math

def tanh(x: float) -> float:
    # Jeżeli x jest bardzo duże lub bardzo małe, tanh dąży do 1 lub -1.
    # Sprawdzamy to, ponieważ np. math.exp(1000) powoduje błąd.
    if x < -100:  return -1
    elif x > 100: return 1

    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        # Zapisujemy wynik tanh do wykorzystania w funkcji backward.
        self.tanh = tensor_apply(tanh, input)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda tanh, grad: (1 - tanh ** 2) * grad,
            self.tanh,
            gradient)

class Relu(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(lambda x: max(x, 0), input)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                              self.input,
                              gradient)

def softmax(tensor: Tensor) -> Tensor:
    """Funkcja softmax dla ostatniego wymiaru"""
    if is_1d(tensor):
        # Zmniejszamy największe wartości dla poprawy stabilności.
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]

        sum_of_exps = sum(exps)                 # To jest suma wag.
        return [exp_i / sum_of_exps             # Prawdopodobieństwo obliczamy jako
                for exp_i in exps]              # część tej sumy
    else:
        return [softmax(tensor_i) for tensor_i in tensor]


class SoftmaxCrossEntropy(Loss):
    """
    To jest ujemny logarytm prawdopodobieństwa obserwowanych danych przy założonym
    modelu sieci neuronowej. Jeżeli wybierzemy wagi tak, aby go zminimalizować, to model
    będzie maksymalizował prawdopodobieństwo obserwowanych danych. 
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # używamy funkcji softmax, by uzyskać prawdopodobieństwa
        probabilities = softmax(predicted)

        # To będzie logarytm z p_i dla właściwej klasy i oraz 0 dla pozostałych klas.
        # Dodajemy niewielką liczbę p, aby uniknąć liczenia log(0).
        likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)

        # Następnie dodajemy wartości ujemne.
        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        # Czy to nie jest przyjemne równanie?
        return tensor_combine(lambda p, actual: p - actual,
                              probabilities,
                              actual)

class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True

    def forward(self, input: Tensor) -> Tensor:
        if self.train:
            # Stwórz maskę zer i jedynek w takim kształcie jak input
            # używając określonego prawdopodobieństwa.
            self.mask = tensor_apply(
                lambda _: 0 if random.random() < self.p else 1,
                input)
            # Pomnóż przez maskę, aby wyłączyć niektóre neurony.
            return tensor_combine(operator.mul, input, self.mask)
        else:
            # W trakcie testowania sieci przeskaluj wartości wejściowe.
            return tensor_apply(lambda x: x * (1 - self.p), input)

    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            # Propaguj gradient tylko wtedy, gdy mask == 1.
            return tensor_combine(operator.mul, gradient, self.mask)
        else:
            raise RuntimeError("używaj funkcji backward tylko podczas trenowania")


#plt.savefig('im/mnist.png')
#plt.gca().clear()

def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels=5) == [0, 0, 1, 0, 0]


from scratch.linear_algebra import squared_distance

import json

def save_weights(model: Layer, filename: str) -> None:
    weights = list(model.params())
    with open(filename, 'w') as f:
        json.dump(weights, f)

def load_weights(model: Layer, filename: str) -> None:
    with open(filename) as f:
        weights = json.load(f)

    # sprawdź zgodność
    assert all(shape(param) == shape(weight)
               for param, weight in zip(model.params(), weights))

    # następnie załaduj
    for param, weight in zip(model.params(), weights):
        param[:] = weight

def main():
    
    # kolejne podejście do bramki XOR
    
    # dane treningowe
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]
    
    random.seed(0)
    
    net = Sequential([
        Linear(input_dim=2, output_dim=2),
        Sigmoid(),
        Linear(input_dim=2, output_dim=1)
    ])
    
    import tqdm
    
    optimizer = GradientDescent(learning_rate=0.1)
    loss = SSE()
    
    with tqdm.trange(3000) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)
    
                optimizer.step(net)
    
            t.set_description(f"xor loss {epoch_loss:.3f}")
    
    for param in net.params():
        print(param)
    
    
    # kolejne podejście do gry Fizz Buzz
    
    from scratch.neural_networks import binary_encode, fizz_buzz_encode, argmax
    
    xs = [binary_encode(n) for n in range(101, 1024)]
    ys = [fizz_buzz_encode(n) for n in range(101, 1024)]
    
    NUM_HIDDEN = 25
    
    random.seed(0)
    
    net = Sequential([
        Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
        Tanh(),
        Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform'),
        Sigmoid()
    ])
    
    def fizzbuzz_accuracy(low: int, hi: int, net: Layer) -> float:
        num_correct = 0
        for n in range(low, hi):
            x = binary_encode(n)
            predicted = argmax(net.forward(x))
            actual = argmax(fizz_buzz_encode(n))
            if predicted == actual:
                num_correct += 1
    
        return num_correct / (hi - low)
    
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)
    loss = SSE()
    
    with tqdm.trange(1000) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)
    
                optimizer.step(net)
    
            accuracy = fizzbuzz_accuracy(101, 1024, net)
            t.set_description(f"fb loss: {epoch_loss:.2f} acc: {accuracy:.2f}")
    
    # teraz sprawdźmy wyniki na zbiorze testowym
    print("test results", fizzbuzz_accuracy(1, 101, net))
    
    random.seed(0)
    
    net = Sequential([
        Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
        Tanh(),
        Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform')
        # teraz bez końcowej warstwy sigmoid
    ])
    
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)
    loss = SoftmaxCrossEntropy()
    
    with tqdm.trange(100) as t:
        for epoch in t:
            epoch_loss = 0.0
    
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)
    
                optimizer.step(net)
    
            accuracy = fizzbuzz_accuracy(101, 1024, net)
            t.set_description(f"fb loss: {epoch_loss:.3f} acc: {accuracy:.2f}")
    
    # ponownie sprawdzamy wyniki na zbiorze testowym
    print("test results", fizzbuzz_accuracy(1, 101, net))
    
    
    # Pobranie danych MNIST 
    
    import mnist
    
    # Ten fragment pobiera dane. Zmień ścieżkę na taką, jaką chcesz.
    # (Tak, to jest funkcja, która nie ma argumentów, tego właśnie oczekuje ta biblioteka).
    # (Tak, przypisuję lambdę do zmiennej, mimo że mówiłem, że nie należy tak robić).
    mnist.temporary_dir = lambda: '/tmp'
    
    # Każda z tych funkcji pobiera dane i zwraca tablicę numpy.
    # Wywołujemy .tolist(), ponieważ nasze "tensory" są tak naprawdę listami.
    train_images = mnist.train_images().tolist()
    train_labels = mnist.train_labels().tolist()
    
    assert shape(train_images) == [60000, 28, 28]
    assert shape(train_labels) == [60000]
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(10, 10)
    
    for i in range(10):
        for j in range(10):
            # Wyświetl każdy obraz jako czarno-biały i ukryj osie.
            ax[i][j].imshow(train_images[10 * i + j], cmap='Greys')
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)
    
    plt.show()
    
    
    # Ładowanie danych testowych MNIST
    
    test_images = mnist.test_images().tolist()
    test_labels = mnist.test_labels().tolist()
    
    assert shape(test_images) == [10000, 28, 28]
    assert shape(test_labels) == [10000]
    
    
    # Centrowanie obrazów
    
    # Policz średnią wartość piksela
    avg = tensor_sum(train_images) / 60000 / 28 / 28
    
    # Centrowanie, skalowanie i spłaszczanie
    train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                    for image in train_images]
    test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                   for image in test_images]
    
    assert shape(train_images) == [60000, 784], "obrazy powinny być spłaszczone"
    assert shape(test_images) == [10000, 784], "obrazy powinny być spłaszczone"
    
    # po centrowaniu wartość każdego piksela powinna być bliska 0
    assert -0.0001 < tensor_sum(train_images) < 0.0001
    
    
    # Kodowanie 1 z n, dane testowe
    
    train_labels = [one_hot_encode(label) for label in train_labels]
    test_labels = [one_hot_encode(label) for label in test_labels]
    
    assert shape(train_labels) == [60000, 10]
    assert shape(test_labels) == [10000, 10]
    
    
    # Pętla trenująca
    
    import tqdm
    
    def loop(model: Layer,
             images: List[Tensor],
             labels: List[Tensor],
             loss: Loss,
             optimizer: Optimizer = None) -> None:
        correct = 0         # Przechowuje liczbę poprawnych przewidywań.
        total_loss = 0.0    # Przechowuje całkowitą stratę.
    
        with tqdm.trange(len(images)) as t:
            for i in t:
                predicted = model.forward(images[i])             # Określ wartości przewidywane.
                if argmax(predicted) == argmax(labels[i]):       # Sprawdź poprawność.
                    correct += 1                                 
                total_loss += loss.loss(predicted, labels[i])    # Oblicz stratę.
    
                # Podczas treningu propaguj wstecznie gradient i zaktualizuj wagi.
                if optimizer is not None:
                    gradient = loss.gradient(predicted, labels[i])
                    model.backward(gradient)
                    optimizer.step(model)
    
                # Zaktualizuj metryki na pasku postępu.
                avg_loss = total_loss / (i + 1)
                acc = correct / (i + 1)
                t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")
    
    
    # Model regresji logistycznej na danych MNIST
    
    random.seed(0)
    
    # Regresja logistyczna to po prostu warstwa liniowa wraz z funkcją softmax
    model = Linear(784, 10)
    loss = SoftmaxCrossEntropy()
    
    # Ten optymalizator wydaje się działać poprawnie
    optimizer = Momentum(learning_rate=0.01, momentum=0.99)
    
    # Trenowanie modelu na danych treningowych
    loop(model, train_images, train_labels, loss, optimizer)
    
    # Testowanie na danych testowych (dlatego brak optymalizatora)
    loop(model, test_images, test_labels, loss)
    
    
    # Głęboka sieć neuronowa na danych MNIST
    
    random.seed(0)
    
    # Nazywamy je, żeby mieć możliwość włączania i wyłączania.
    dropout1 = Dropout(0.1)
    dropout2 = Dropout(0.1)
    
    model = Sequential([
        Linear(784, 30),  # Ukryta warstwa 1: rozmiar 30
        dropout1,
        Tanh(),
        Linear(30, 10),   # Ukryta warstwa 2: rozmiar 10
        dropout2,
        Tanh(),
        Linear(10, 10)    # Warstwa wyjściowa: rozmiar 10
    ])
    
    
    # Trenowanie modelu głębokiego na danych MNIST
    
    optimizer = Momentum(learning_rate=0.01, momentum=0.99)
    loss = SoftmaxCrossEntropy()
    
    # Włącz warstwę dropout i rozpocznij trening (na moim laptopie zajęło to ponad 20 minut!)
    dropout1.train = dropout2.train = True
    loop(model, train_images, train_labels, loss, optimizer)
    
    # Wyłącz warstwę dropout i przetestuj
    dropout1.train = dropout2.train = False
    loop(model, test_images, test_labels, loss)
    
if __name__ == "__main__": main()
