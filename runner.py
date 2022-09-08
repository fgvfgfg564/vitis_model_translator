from .yacc import yacc


class Runner:
    def __init__(self, config_filename):
        with open(config_filename, "r") as f:
            self.prog = f.read()
        self.ast = yacc.parse(self.prog)
        self.reset()

    def reset(self):
        self.varList = dict()
        self.modules = dict()
        self.feeds = None
        self.inputs = None
        self.outputs = None

    def run(self, inputs):
        self.feeds = inputs
        self.ast.run(self)

        if isinstance(inputs, dict):
            result = dict()
            for each in self.outputs:
                result.setdefault(each, self.varList[each].value)
        else:
            result = []
            for each in self.outputs:
                if self.varList[each] is not None:
                    result.append(self.varList[each].value)
                else:
                    result.append(None)
        return result
