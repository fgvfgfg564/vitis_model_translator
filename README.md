# TND file format
## Definitions
### input x1, x2, ...;
Indicates the inputs of network in order. Should only occur once.
### output x1, x2, ...;
Indicates the outputs of network in order. Should only occur once.
### placeholder x1, x2, ...;
Indicates the internal results of network. 
### module m(n_in, n_out);
Defines a module with `n_in` inputs and `n_out` outputs. The module name `m` here must be the same as **the name corresponding attribute** in Keras Model object. Beware that this name might not be the same as the TensorFlow name of your module.

For example, consider defining a submodule like this in a `Keras.Model` object:
```
    self.encoder = keras.Sequential(..., name="MainEncoder")
```
Then the module name `m` here should be "encoder", not "MainEncoder".

If `n_in` is 1, the module takes a single `tf.Tensor` input.

If `n_in` is greater than 1, the module takes a tuple of multiple `tf.Tensor`s as input.

If `n_out` is 1, the module produces a single `tf.Tensor` output.

If `n_out` is greater than 1, the module produces a tuple of multiple `tf.Tensor`s as output.



## Data flow
### calib m(x1, x2, ...);
Use data x1, x2, ..., to calibrate module m. The number of input tensors must be correspondent to module's iput.

Module must be calibrated before use.

If "calib" is set to None during translating, in this operation, translator simply quantizes the weights without calibrating on data.

### y1, y2, ... yk = split(x)
This operation splits variable `x` into k patches on default dimension `3` and assign them to variable `y1`, `y2`, ..., `yk`.

### y1, y2, ... = m(x1, x2, ...);
Defines a data flow: variable ys are the results of applying module m on variable xs.


## Comments
Use a single character '#' to mark a comment line.

## Examples
```
# An simple example on "Balle 2018" gaussian conditional model:
input x;
output x_recon;
placeholder t0, t1;
module g_a(1, 1);
module g_s(1, 1);
module h_a(1, 1);
module h_s(1, 1);

calib g_a(x);
t0 = g_a(x);
calib g_s(t0);
x_recon = g_s(t0);
calib h_a(t0);
t1 = h_a(t0);
calib h_s(t1);
```

# Inputs on translator
### calib_dataset
`tf.data.Dataset` or a tuple consisting numpy arrays with the same dimension 0 corresponding to the defined 'input' variables. The suggested dataset length is between 100 and 1000.

Sometimes an oversized calibration dataset causes memory overflow. Consider decreasing the dataset size to solve this problem.