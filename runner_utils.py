from ctypes import *
from typing import List
import numpy as np
try:
    import vart
except ImportError as e:
    pass
import xir


def get_child_subgraph_dpu(graph):
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


class FixPosArray:
    def __init__(self, array: np.ndarray, fixpoint: int = None) -> None:
        self.array = array
        if array.dtype == np.int8 and fixpoint is None:
            fixpoint = 0
        self.fixpoint = fixpoint

    def changeFixPoint(self, newfixpoint: int):
        if self.fixpoint == None:
            # Now self.array is in float-point format
            self.array = np.clip((self.array * (2**newfixpoint)), -128, 127).astype(
                np.int8
            )
        else:
            # Now just change the fix point value
            if newfixpoint == self.fixpoint:
                return
            if self.fixpoint < newfixpoint:
                self.array = np.left_shift(self.array, newfixpoint - self.fixpoint)
            else:
                self.array = np.right_shift(self.array, self.fixpoint - newfixpoint)
            self.fixpoint = newfixpoint

    @property
    def value(self):
        return self.array.astype(np.float64) / (2**self.fixpoint)

    @property
    def shape(self):
        return self.array.shape


class ModuleRunner:
    def __init__(self, model_filename) -> None:
        self.model = xir.Graph.deserialize(model_filename)
        self.subgraphs = get_child_subgraph_dpu(self.model)
        assert len(self.subgraphs) == 1
        self.runner = vart.Runner.create_runner(self.subgraphs[0], "run")
        self.inputTensors = self.runner.get_input_tensors()
        self.outputTensors = self.runner.get_output_tensors()
        self.n_in = len(self.inputTensors)
        self.n_out = len(self.outputTensors)

        self.outputBuffer = list(
            [
                np.empty(tuple(x.dims), dtype=np.int8, order="C")
                for x in self.outputTensors
            ]
        )

    def run(self, inputs):
        for i, each in enumerate(inputs):
            if tuple(each.shape) != tuple(self.inputTensors[i].dims):
                raise ValueError(
                    f"Shapes mismatch: {tuple(each.shape)}(received) vs. {tuple(self.inputTensors[i].dims)}(expected)")
            each.changeFixPoint(self.getInputFixPos(i))
        input_arrays = list([x.array for x in inputs])
        job_id = self.runner.execute_async(
            input_arrays, self.outputBuffer, enable_dynamic_array=True)
        self.runner.wait(job_id)

    def getInputFixPos(self, idx):
        return self.inputTensors[idx].get_attr("fix_point")

    def getOutputFixPos(self, idx):
        return self.outputTensors[idx].get_attr("fix_point")

    def __call__(self, x):
        return self.run(x)

    @property
    def outputs(self):
        return list(
            [
                FixPosArray(self.outputBuffer[i], self.getOutputFixPos(i))
                for i in range(self.n_out)
            ]
        )

    def execute(self, inputs):
        self.run(inputs)
        return self.outputs
