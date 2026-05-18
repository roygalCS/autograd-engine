from nn import MLP
from Utils import draw_dot


# fake net / data
model = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# one forward pass
ypred = [model(x) for x in xs]

# MSE loss func
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

loss.backward()

dot = draw_dot(loss)
dot.render('my_network_graph', format='png', view=True)
