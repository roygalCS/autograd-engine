class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: none
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1 * output.grad
            other.grad += 1 * output.grad
        out._backward = _backward

        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return output

    def __rmul__(self, other):
        return self * other

    def exp(self):
        x = self.data
        out = Value(x.exp(), (self, ), 'exp')

        def _backward(self):
            self.grad += out.data * out.grad
        out._backward = backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self / other**-1

    def pow(self, other):
        assert isinstance(other, (int, float)
                          ), "for now only supporting int and float exponent"
        out = Value(self.data**other, (self, other), f'**{other}')

        def _backward(self, other):
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1 / math.exp(2*x)+1)
        out = Value(t, (self, ), label='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            node._backward()
