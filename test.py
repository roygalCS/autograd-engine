from engine import Value
from Utils import draw_dot
from nn import MLP

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a + b

dot = draw_dot(c)
dot.render('my_graph', format='png', view=True)
