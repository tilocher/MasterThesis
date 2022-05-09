# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
import sys
import os
nn_dir = os.path.abspath(os.path.join(__file__, os.pardir))
base_dir = os.path.abspath(os.path.join(nn_dir, os.pardir))
sys.path.append(nn_dir)
sys.path.append(base_dir)