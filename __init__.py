try:
    from .translator import Translator, Deployer

    print("Translator mode activated.")
except ModuleNotFoundError:
    pass

try:
    from .runner import Runner

    print("Runner mode activated.")
except ModuleNotFoundError:
    pass
