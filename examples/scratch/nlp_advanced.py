from scratch.deep_learning import Optimizer, Layer

class EmbeddingOptimizer(Optimizer):
    """
    Zoptymalizowane dla przypadku, gdzie mamy tylko
    wartstwy embeding z aktualizacją pojedynczych identyfikatorów
    """
    def __init__(self, learning_rate: float) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Znajdź pierwszy rekord z niezerowymi wartościami.
            for idx, row in enumerate(grad):
                if row[0] != 0:
                    break

            # Zaktualizuj ten rekord.
            for j in range(len(row)):
                param[idx][j] -= grad[idx][j] * self.lr
