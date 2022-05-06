from curses import use_default_colors
from .yacc import yacc
import tensorflow as tf
import numpy as np


class Translator:
    def __init__(self, config_filename):
        with open(config_filename, "r") as f:
            self.prog = f.read()
        self.ast = yacc.parse(self.prog)

    def translate(self, model, calib_dataset=None, calib_steps=None, init_quant=False):
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
        self.ast.proc(self)

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

class Deployer:
    def __init__(self, config_filename):
        with open(config_filename, "r") as f:
            self.prog = f.read()
        self.ast = yacc.parse(self.prog)
    
    def deploy(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        return self.ast.deploy(self)