import os
import numpy as np
from logging import INFO, log
from tqdm import tqdm

RUNNERMODE = False

try:
    import tensorflow as tf
    from .translator_utils import dataset2np, get_quantized_model, deploy_qat_model
except ModuleNotFoundError:
    RUNNERMODE = True

if RUNNERMODE:
    from .runner_utils import FixPosArray, ModuleRunner


class ASTNodeBase:
    def proc(self, translator):
        raise NotImplementedError

    def run(self, runner):
        raise NotImplementedError

    def crossPlatformProcess(self, engine):
        raise NotImplementedError

    def deploy(self, deployer):
        pass


class Program(ASTNodeBase):
    def __init__(self, expr):
        self.exprs = [expr]

    def append(self, expr):
        self.exprs.append(expr)

    def proc(self, translator):
        if RUNNERMODE:
            raise NotImplementedError(
                "Quantize operations are unavailable in RUNNER MODE!"
            )
        for each in self.exprs:
            each.proc(translator)

    def run(self, runner):
        for each in self.exprs:
            each.run(runner)

    def deploy(self, deployer):
        for each in self.exprs:
            each.deploy(deployer)


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
                    input_dataset = dataset2np(input_dataset, len(self.varList))
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

    def deploy(self, deployer):
        if self.type == "input":
            for i, each in enumerate(self.varList):
                deployer.varShapes.setdefault(each, deployer.input_shape[i])
        else:
            for i, each in enumerate(self.varList):
                deployer.varShapes.setdefault(each, None)

    def run(self, runner):
        for each in self.varList:
            runner.varList.setdefault(each, None)
        if self.type == "input":
            runner.inputs = self.varList
            if isinstance(runner.feeds, dict):
                runner.varList.update(runner.feeds)
            else:
                for k, v in zip(self.varList, runner.feeds):
                    runner.varList[k] = FixPosArray(v)
        if self.type == "output":
            runner.outputs = self.varList


class ModuleDefinition(ASTNodeBase):
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out

    def proc(self, translator):
        translator.checkModuleNotDefined(self.name)
        translator.moduleDefs.setdefault(self.name, (self.n_in, self.n_out))

    def run(self, runner):
        module_filename = self.name + ".xmodel"
        module_runner = ModuleRunner(module_filename)
        print(f"runner created from file: {module_filename}")
        runner.modules.setdefault(self.name, module_runner)


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

    def run(self, runner):
        pass

    def deploy(self, deployer):
        module = deployer.model.__getattribute__(self.name)
        input_spec_strs = []
        for i, each in enumerate(self.varlist):
            input_shape = deployer.varShapes[each]
            input_shape_str = ",".join([str(x) for x in input_shape])
            input_spec_str = f'"{module.inputs[i].name}": "{input_shape_str}"'
            input_spec_strs.append(input_spec_str)
        input_shape_option = '"input_shape": {%s}' % (", ".join(input_spec_strs))
        option = "--options '{%s}'" % input_shape_option
        output_dir = deployer.output_dir
        deploy_qat_model(module, os.path.join(output_dir, self.name + ".h5"))
        with open(os.path.join(output_dir, self.name + ".opt"), "w") as f:
            print(option, file=f)


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
        results = dict()
        for x in tqdm(input_dataset):
            y = module(x)
            for i, each in enumerate(self.target):
                val = (y[i] if len(self.target) > 1 else y).numpy()
                results.setdefault(each, [])
                results[each].append(val)
        for i, each in enumerate(self.target):
            translator.varList[each] = np.concatenate(results[each], axis=0)

    def run(self, runner):
        inputs = [runner.varList[name] for name in self.source]
        outputs = runner.modules[self.function].execute(inputs)
        for k, v in zip(self.target, outputs):
            runner.varList[k] = v

    def deploy(self, deployer):
        module = deployer.model.__getattribute__(self.function)
        inputs = tuple([np.zeros(deployer.varShapes[x]) for x in self.source])
        outputs = module(inputs)
        if len(self.target) == 1:
            outputs = [outputs]
        for i, each in enumerate(self.target):
            deployer.varShapes[each] = outputs[i].shape


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

    def run(self, runner):
        self.crossPlatformProcess(runner)

    def deploy(self, deployer):
        inputs = np.zeros(deployer.varShapes[self.source])
        num_splits = len(self.target)
        splits = np.split(inputs, num_splits, axis=3)
        if len(self.target) == 1:
            outputs = [outputs]
        for i, each in enumerate(self.target):
            deployer.varShapes[each] = outputs[i].shape
