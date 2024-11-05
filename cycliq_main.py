import numpy as np
from cycliq_class import CycLiq
import matplotlib.pyplot as plt


# Initial condition
c_ein = 0.833 # void ratio
c_pin = 1000 #kPa # mean effective stress


# calculate
cycliq_1ele = CycLiq(np.identity(3)*c_pin, c_ein)

dstrain = 1e-5
maxstrain = 0.01
n = int(maxstrain/dstrain)
#n = 100
print('Number of steps:',n)
dStrain = np.zeros((3,3))
dStrain[0,0] = dstrain
dStrain[1,1] = -dstrain*0.5 # undrained
dStrain[2,2] = -dstrain*0.5
result = [[0,c_pin,0]] #[eps1,p,q]
print('# , substeps, axial strain, p, q')

'''
cycliq_1ele.set_next_Strain(dStrain)
cycliq_1ele.calc_next_step()
'''
for i in range(n):
  #cycliq_1ele.set_next_Strain(dStrain)
  cycliq_1ele.calc_mainstep(dStrain)
  t_eps1 = 100*cycliq_1ele.Strn_now[0,0]
  t_p = cycliq_1ele.p_now
  t_Strs_dev = cycliq_1ele.Strs_dev_now.copy()
  t_q = (1.5*np.tensordot(t_Strs_dev,t_Strs_dev))**0.5
  result.append([t_eps1,t_p,t_q])
  print('#',i,';',cycliq_1ele.nsub,';',result[-1])
  '''
  if i%100==0:
    print('#',i,';',cycliq_1ele.nsub,';',result[-1])
  '''


# visualization
result = np.array(result)
fig, axes = plt.subplots(2, 1, figsize=(4, 8), squeeze=False, tight_layout=True)
plt.ticklabel_format(style='plain')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

axes[0,0].set_xlabel(r"$\epsilon_{xx}$(%)")
axes[0,0].set_ylabel(r"q")
axes[0,0].plot(result[:,0], result[:,2])
axes[0,0].plot(result[0,0], result[0,2],marker='o')

axes[1,0].set_xlabel(r"p")
axes[1,0].set_ylabel(r"q")
axes[1,0].plot(result[:,1], result[:,2])
axes[1,0].plot(result[0,1], result[0,2],marker='o')

#plt.savefig("filename.png")
plt.show()
