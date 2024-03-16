"""my_controller_iCreate controller."""

from driver import webotsDriver
from create3 import create3Driver
import numpy as np
import matplotlib.pyplot as plot
np.random.seed(0)

# Create a robot instance from either the webotsDriver or the create3Driver class
bot = webotsDriver()

def run_bot(mode, context=0, randomize=True):
    if mode not in ["learn_context", "learn_path", "exploit"]:
        raise ValueError("Invalid mode. Choose either 'learn_context', 'learn_path', or 'exploit'")

    if mode == "learn_context":
        bot.clear()
        bot.initialization(context, mode="learning", randomize=randomize)
        bot.run("explore")
    elif mode == "learn_path":
        bot.initialization(context, mode="dmtp", randomize=randomize)
        bot.run("explore")
    else: # mode == "exploit"
        bot.initialization(context, mode="dmtp", randomize=randomize)
        bot.run("exploit")

# CLOSE ALL FIGURES AND WAIT FOR WORLD TO RELOAD IN BETWEEN RUNS
# PICKLE FILES SHOULD APPEAR AFTER learn_path

run_bot("exploit")