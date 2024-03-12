import matplotlib.pyplot as plot
from matplotlib_venn import venn2

# Use the venn2 function
venn2(subsets = (555, 664, 551), set_labels = ('Normal skewness', 'Normal kurtosis'))
plot.show()