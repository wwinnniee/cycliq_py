import numpy as np
from cycliq_class import CycLiq
import matplotlib.pyplot as plt
import pprint


# Initial condition
Dr = 0.6
c_emax = 0.973
c_emin = 0.635
c_ein = -Dr*(c_emax-c_emin)+c_emax # void ratio
c_pin = 100 #kPa # mean effective stress


# calculate
cycliq_1ele = CycLiq(np.identity(3)*c_pin, c_ein)

dstrain = 1e-7
dStrain = np.zeros((3,3))
'''
dStrain[0,0] = dstrain
dStrain[1,1] = -dstrain*0.5 # undrained
dStrain[2,2] = -dstrain*0.5
'''
dStrain[0,1] = dstrain
result = [[0,c_pin,0]] #[eps1,p,tau]
print('# , substeps, axial strain, p, tau')

nreverse = 20
ireverse = 0
istep = 0
maxtau = 25 #kPa
while ireverse<nreverse:
  cycliq_1ele.calc_mainstep(dStrain)
  t_eps1 = 100*cycliq_1ele.Strn_now[0,1]
  t_p = cycliq_1ele.p_now
  t_Strs_dev = cycliq_1ele.Strs_dev_now.copy()
  #t_q = (1.5*np.tensordot(t_Strs_dev,t_Strs_dev))**0.5
  t_tau = cycliq_1ele.Strs_now[0,1]
  result.append([t_eps1,t_p,t_tau])
  if istep%1000==0:
    with open(f'./cycliq_undrained-cyclic_output{istep}.txt', 'w') as f:
      pprint.pprint(vars(cycliq_1ele), stream=f)
  '''
  if istep%10000==0:
    print('#',istep,';',cycliq_1ele.nsub,';',result[-1])
  '''
  #if t_q>maxq:
  if abs(t_tau)>maxtau:
    dStrain *= -1
    ireverse += 1
    print('Loading reversed :', ireverse)
    with open(f'./cycliq_undrained-cyclic_output{istep}.txt', 'w') as f:
      pprint.pprint(vars(cycliq_1ele), stream=f)
    #print('At r =', cycliq_1ele.r_now)
  istep += 1
print('nstep =', istep)


# visualization
result = np.array(result)
fig, axes = plt.subplots(2, 1, figsize=(4, 8), squeeze=False, tight_layout=True)
plt.ticklabel_format(style='plain')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

axes[0,0].set_xlabel(r"$\epsilon_{xx}$(%)")
axes[0,0].set_ylabel(r"$\tau$")
axes[0,0].plot(result[:,0], result[:,2])
axes[0,0].plot(result[0,0], result[0,2],marker='o')

axes[1,0].set_xlabel(r"p")
#axes[1,0].set_ylabel(r"q")
axes[0,0].set_ylabel(r"$\tau$")
axes[1,0].plot(result[:,1], result[:,2])
axes[1,0].plot(result[0,1], result[0,2],marker='o')

#plt.savefig("filename.png")
plt.show()
