import logging
import warnings

def silence_noise():
    for name in ["matplotlib", "pymc", "aesara", "numba", "arviz", "urllib3", "streamlit"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
