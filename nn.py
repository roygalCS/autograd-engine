from engine import Value


class Neuron:

    def __init__(self, nin):
        self.w = [(Value(random.uniform(-1, 1))) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


class Layer:

    # params is (dimension, neuron count per dimension)
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def _call_(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 0 else outs


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
