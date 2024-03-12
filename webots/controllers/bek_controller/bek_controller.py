"""my_controller_iCreate controller."""

from driver import Driver
import numpy as np
import matplotlib.pyplot as plot
np.random.seed(0)
bot = Driver()

# CLOSE ALL FIGURES AND WAIT FOR WORLD TO RELOAD IN BETWEEN RUNS
# PICKLE FILES SHOULD APPEAR AFTER SECOND STEP

# THIS SHOULD BE UNCOMMENTED FOR THE FIRST RUN ONLY
#bot.clear()

# STEP1: mode="learning", bot.run("explore")
# STEP2: mode="dmtp", bot.run("explore")
# STEP3: mode="dmtp", bot.run("exploit")
bot.initialization(0, mode="dmtp", randomize=True)
# bot.visualize_replay(4, 'all')
bot.run("exploit")

## TODO:
## Go phenomelogical
## 1. Multiscale
## 2. Adaptive Scales
## 3. Intentional exploration
