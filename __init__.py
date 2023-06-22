import logging

try:
    from .translator import Translator, Deployer

    logging.info("Translator mode activated.")
except ModuleNotFoundError:
    pass

try:
    from .runner import Runner

    logging.info("Runner mode activated.")
except ModuleNotFoundError:
    pass
