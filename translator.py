from curses import use_default_colors

import numpy as np
import tensorflow as tf

from .yacc import yacc


class ASTFunctionBase:
    def __init__(self, config_filename):
        with open(config_filename, "r") as f:
            self.prog = f.read()
        self.ast = yacc.parse(self.prog)


class Translator(ASTFunctionBase):
    def translate(self, model, calib_dataset=None, calib_steps=None, init_quant=False, use_default=False):
        self.reset()
        self.model = model
        if calib_dataset is None:
            self.calib_dataset = None
        else:
            self.calib_dataset = []
            if isinstance(calib_dataset, tf.data.Dataset):
                calib_dataset = [calib_dataset]
            for each in calib_dataset:
                if isinstance(each, tf.data.Dataset):
                    each = np.concatenate(
                        list(each.take(calib_steps).as_numpy_iterator()),
                        axis=0,
                    )[:calib_steps]
                self.calib_dataset.append(each)
        self.calib_steps = calib_steps
        self.init_quant = init_quant
        self.use_default = use_default
        self.ast.proc(self)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.keras.backend.clear_session()
        tf.keras.backend.clear_session()

    def reset(self):
        self.inputs = None
        self.outputs = None
        self.varList = dict()
        self.moduleDefs = dict()

    def checkVarNameDefined(self, name):
        if name not in self.varList:
            raise ValueError(f"Undefined symbol: {name}")

    def checkVarNameNotDefined(self, name):
        if name in self.varList:
            raise ValueError(f"Redundant variable definition: {name}")

    def checkModuleDefined(self, name):
        if name not in self.moduleDefs:
            raise ValueError(f"Undefined module: {name}")

    def checkModuleNotDefined(self, name):
        if name in self.moduleDefs:
            raise ValueError(f"Redundant module definition: {name}")

    def checkModuleInput(self, module, n_in):
        self.checkModuleDefined(module)
        required = self.moduleDefs[module][0]
        if required != n_in:
            raise ValueError(f"Module {module} requires {required} inputs. Got {n_in}.")

    def checkModuleOutput(self, module, n_out):
        self.checkModuleDefined(module)
        required = self.moduleDefs[module][1]
        if required != n_out:
            raise ValueError(
                f"Module {module} requires {required} outputs. Got {n_out}."
            )

    def makeDataset(self, vars, use_tf=False):
        ds = []
        for each in vars:
            self.checkVarNameDefined(each)
            value = self.varList.get(each)
            ds.append(value)
        ds = tuple(ds)
        if len(vars) == 1:
            ds = ds[0]
        if use_tf:
            ds = tf.data.Dataset.from_tensor_slices(ds)
            ds = ds.batch(1)
        return ds


class Deployer(ASTFunctionBase):
    def reset(self):
        self.varShapes = dict()

    def deploy(self, model, output_dir, input_shape):
        self.reset()
        self.model = model
        self.output_dir = output_dir
        self.input_shape = input_shape
        return self.ast.deploy(self)
