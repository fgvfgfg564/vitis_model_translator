import numpy as np
from logging import INFO, log
RUNNERMODE = False

try:
    import tensorflow as tf
    from .quant_utils import dataset2np, get_quantized_model
except ModuleNotFoundError:
    RUNNERMODE = True
    print("Tensorflow not detected. The translator is now in RUNNER MODE.")

if RUNNERMODE:
    from runner import FixPosArray, Runner


class ASTNodeBase:
    def proc(self, translator):
        raise NotImplementedError

    def run(self, superRunner):
        raise NotImplementedError

    def crossPlatformProcess(self, engine):
        raise NotImplementedError


class Program(ASTNodeBase):
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


class VarDefinition(ASTNodeBase):
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

    def run(self, superRunner):
        for each in self.varList:
            superRunner.varList.setdefault(each, None)
        if self.type == "input":
            superRunner.inputs = self.varList
            if isinstance(superRunner.feeds, dict):
                for k, v in superRunner.feeds.items():
                    superRunner.varList[k] = FixPosArray(v)
        if self.type == "output":
            superRunner.outputs = self.varList


class ModuleDefinition(ASTNodeBase):
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out

    def proc(self, translator):
        translator.checkModuleNotDefined(self.name)
        translator.moduleDefs.setdefault(self.name, (self.n_in, self.n_out))

    def run(self, superRunner):
        module_filename = self.name + ".xmodel"
        runner = Runner(module_filename)
        superRunner.modules.setdefault(self.name, runner)


class Calib(ASTNodeBase):
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

    def run(self, superRunner):
        pass


class Assign(ASTNodeBase):
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

    def run(self, superRunner):
        inputs = [superRunner.varList[name] for name in self.source]
        outputs = superRunner.modules[self.function].execute(inputs)
        for k, v in zip(self.target, outputs):
            superRunner.varList[k] = v


class Split(ASTNodeBase):
    def __init__(self, target, source):
        self.target = target
        self.source = source

    def crossPlatformProcess(self, engine):
        num_splits = len(self.target)
        splits = np.split(engine.varList.get(self.source), num_splits, axis=3)

        for i, each in enumerate(self.target):
            engine.varList.set(each, splits[i])

    def proc(self, translator):
        translator.checkVarNameDefined(self.source)
        for each in self.target:
            translator.checkVarNameDefined(each)

        self.crossPlatformProcess(translator)

    def run(self, superRunner):
        self.crossPlatformProcess(superRunner)
