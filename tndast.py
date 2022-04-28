import numpy as np
from logging import INFO, log
RUNNERMODE = False

try:
    import tensorflow as tf
    from .quant_utils import dataset2np, get_quantized_model
except ModuleNotFoundError:
    RUNNERMODE = True
    print("Tensorflow not detected. The translator is now in RUNNER MODE.")


class Program:
    def __init__(self, expr):
        self.exprs = [expr]

    def append(self, expr):
        self.exprs.append(expr)

    def proc(self, translator):
        if RUNNERMODE:
            raise NotImplementedError(
                "Quantize operations are unavailable in RUNNER MODE!")
        for each in self.exprs:
            each.proc(translator)

    def run(self, superRunner):
        for each in self.exprs:
            each.run(superRunner)


class VarDefinition:
    def __init__(self, type, varList):
        self.type = type
        self.varList = varList

    def proc(self, translator):
        for each in self.varList:
            translator.checkVarNameNotDefined(each)
        if self.type == "input":
            if translator.inputs is not None:
                raise SyntaxError("Detected multiple input definitions.")
            translator.inputs = self.varList
            if translator.init_quant:
                if len(translator.calib_dataset) != len(self.varList):
                    raise ValueError(
                        "Length of calib dataset must be correspondent with defined 'input' variables. Found {} vs. {}.".format(
                            len(translator.calib_dataset), len(self.varList)
                        )
                    )
                input_dataset = translator.calib_dataset
                if isinstance(input_dataset, tf.data.Dataset):
                    input_dataset = dataset2np(
                        input_dataset, len(self.varList))
                for i, each in enumerate(self.varList):
                    translator.varList.setdefault(each, input_dataset[i])
            else:
                for i, each in enumerate(self.varList):
                    translator.varList.setdefault(each, None)
        else:
            if self.type == "output":
                if translator.outputs is not None:
                    raise SyntaxError("Detected multiple output definitions.")
                translator.outputs = self.varList
            for each in self.varList:
                translator.varList.setdefault(each, None)


class ModuleDefinition:
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out

    def proc(self, translator):
        translator.checkModuleNotDefined(self.name)
        translator.moduleDefs.setdefault(self.name, (self.n_in, self.n_out))


class Calib:
    def __init__(self, name, varlist) -> None:
        self.name = name
        self.varlist = varlist

    def proc(self, translator):
        translator.checkModuleInput(self.name, len(self.varlist))
        module = translator.model.__getattribute__(self.name)
        calib_dataset = None
        if translator.init_quant:
            calib_dataset = translator.makeDataset(self.varlist)

        print(f"Calibrating module '{self.name}'")
        quant_module = get_quantized_model(
            module,
            calib_dataset=calib_dataset,
            calib_steps=translator.calib_steps,
            init_quant=translator.init_quant,
        )
        translator.model.__setattr__(self.name, quant_module)


class Assign:
    def __init__(self, target, function, source) -> None:
        self.target = target
        self.function = function
        self.source = source

    def proc(self, translator):
        translator.checkModuleDefined(self.function)
        for each in self.target:
            translator.checkVarNameDefined(each)
        for each in self.source:
            translator.checkVarNameDefined(each)
        translator.checkModuleInput(self.function, len(self.source))
        translator.checkModuleOutput(self.function, len(self.target))

        if not translator.init_quant:
            return

        module = translator.model.__getattribute__(self.function)

        input_dataset = translator.makeDataset(self.source, use_tf=True)
        for each in self.target:
            translator.varList[each] = None
        for x in input_dataset:
            y = module(x)
            for i, each in enumerate(self.target):
                val = (y[i] if len(self.target) > 1 else y).numpy()
                if translator.varList[each] is None:
                    translator.varList[each] = val
                else:
                    a = translator.varList[each]
                    translator.varList[each] = np.concatenate((a, val), axis=0)


class Split:
    def __init__(self, target, source):
        self.target = target
        self.source = source

    def proc(self, translator):
        translator.checkVarNameDefined(self.source)
        for each in self.target:
            translator.checkVarNameDefined(each)

        num_splits = len(self.target)
        splits = np.split(translator.varList.get(
            self.source), num_splits, axis=3)

        for i, each in enumerate(self.target):
            translator.varList.set(each, splits[i])
