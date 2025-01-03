import numpy as np
from cycliq_class import CycLiq
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import glob
import pprint
import traceback

#output path
output_path = './output_debug' # without '/' at the end

# Remove previous files
if True:
  for f in glob.glob(f'{output_path}/*.txt'):
    os.remove(f)

# inputs
'''
stress_pre
c_ein_

a_dstrn_vol
a_dStrn_dev
strn_vol_ir_pre
strn_vol_re_pre
strn_vol_c_0_
strn_vol_c_pre
r_alpha
M_max
strn_vol_ir_reversal
gammamono
strn_vol_pre

psi
'''
noconvergence_vars = \
[1.9463639333355331e+00, \
2.1931689585171030e-03, \
0.0000000000000000e+00, \
2.1931689585171030e-03, \
1.9330681600016817e+00, \
0.0000000000000000e+00, \
0.0000000000000000e+00, \
0.0000000000000000e+00, \
1.9386211234182049e+00, \
7.7020000000000000e-01, \
-2.1740897315154223e-09, \
4.6182643950729379e-09, \
-1.2781687781281859e-09, \
-0.0000000000000000e+00, \
-1.2781687781281859e-09, \
-5.3429609722447447e-09, \
-0.0000000000000000e+00, \
-0.0000000000000000e+00, \
-0.0000000000000000e+00, \
7.2469657717180740e-10, \
5.0241679306232349e-08, \
0.0000000000000000e+00, \
-6.1651526294364558e-04, \
-1.3803874830925580e-08, \
-3.9611660077767774e-01, \
6.8554239141603345e-03, \
0.0000000000000000e+00, \
6.8554239141603345e-03, \
7.9223320155535559e-01, \
0.0000000000000000e+00, \
0.0000000000000000e+00, \
0.0000000000000000e+00, \
-3.9611660077767774e-01, \
1.2008360307868460e+00, \
0.0000000000000000e+00, \
5.9449801706781853e-07, \
6.4176726300741709e-06, \
-1.6262574126565943e-01]



a_Strs_pre = np.array(noconvergence_vars[0:9]).reshape(3,3)
a_c_ein = noconvergence_vars[9]

t_dstrn_vol = noconvergence_vars[10]
t_dStrn_dev = np.array(noconvergence_vars[11:20]).reshape(3,3)
a_dStrain = t_dStrn_dev + t_dstrn_vol*np.identity(3)

debug_vars = noconvergence_vars[20:]

# calculate
cycliq_1ele = CycLiq(\
  a_Strs_in=a_Strs_pre, a_c_ein=a_c_ein, \
  DEBUG=True, debug_vars=debug_vars)

cycliq_1ele.calc_mainstep(a_dStrain)