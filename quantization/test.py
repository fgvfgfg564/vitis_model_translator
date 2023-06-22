from preprocessing import *
import tensorflow
from tensorflow import keras
from keras.layers import *
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

# model2 = add_l2_w_a_loss(model, 0.001)
model2.summary()
# print(model2.losses)

# model3 = remove_l2_reg_capture(model2)
# model3.summary()

# for layer in model.layers:
#     weights = layer.trainable_weights
#     for weight in weights:
#         print(K.sum(weight))