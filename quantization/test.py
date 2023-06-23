import tensorflow
from keras.layers import *
from preprocessing import *
from tensorflow import keras
from utils import *

d1 = Dense(10, name='branch1')
d2 = Dense(20, name='branch2')
con = Concatenate()

a = Input([10])
x1 = d1(a)
x2 = d2(a)
b = con([x1, x2])
model = keras.Model(a,b)

print(model.losses)

filter = lambda layer: isinstance(layer, Dense)
model2 = insert_layer_nonseq(model, filter, lambda layer: Dense(30, name=layer.name+"extra"))

model2.summary()