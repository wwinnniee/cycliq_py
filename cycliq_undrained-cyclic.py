import numpy as np
from cycliq_class import CycLiq
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import os
import glob
import pprint
import traceback

def plot(ifShow=False):
  rslt = np.array(result)
  fig, axes = plt.subplots(2, 1, figsize=(6, 5), squeeze=False, tight_layout=True)
  plt.ticklabel_format(style='plain')
  plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
  plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

  axes[0,0].set_xlabel(r"$\epsilon_{xy}$")
  axes[0,0].set_ylabel(r"$\tau$")
  axes[0,0].set_xlim(-0.2,0.4)
  axes[0,0].set_ylim(-40,40)
  axes[0,0].plot(rslt[:,0], rslt[:,2])
  axes[0,0].plot(rslt[0,0], rslt[0,2],marker='o')

  axes[1,0].set_xlabel(r"p")
  axes[1,0].set_ylabel(r"$\tau$")
  axes[1,0].set_xlim(0,c_pin)
  axes[1,0].set_ylim(-40,40)
  axes[1,0].plot(rslt[:,1], rslt[:,2])
  axes[1,0].plot(rslt[0,1], rslt[0,2],marker='o')

  '''
  axes[2,0].set_xlabel(r"$\tau$")
  axes[2,0].set_ylabel(r"L, $\epsilon_{re}$, $\epsilon_{ir}$")
  #axes[2,0].plot(rslt[:,2], rslt[:,4], label=r'$\epsilon_{re}$')
  #axes[2,0].plot(rslt[:,2], rslt[:,5], label=r'$\epsilon_{ir}$')
  axes[2,0].plot(rslt[:,2], rslt[:,3], label='L')
  '''
  '''
  axes20 = axes[2,0].twinx()
  axes20.plot(rslt[:,2], rslt[:,3], label='L')
  axes[2,0].legend()
  axes20.legend()
  '''
  '''
  axes[2,0].set_xlabel(r"istep")
  axes[2,0].set_ylabel(r"p")
  axes[2,0].plot(range(len(rslt[:,1])), rslt[:,1])
  axes[2,0].plot(rslt[0,1], rslt[0,1],marker='o')
  '''
  plt.savefig(f"{output_path}/plot_{ireverse}_{istep}.png")
  if ifShow:
    plt.show()
  plt.close()



#output path
output_path = './output_undrained-cyclic' # without '/' at the end

# Conditions
nreverse = 44
dstrain = 4e-6
c_pin = 100 #kPa # mean effective stress
maxtau = 25 #kPa
Dr = 0.466



# Remove previous files
if True:
  for f in glob.glob(f'{output_path}/vars*.txt'):
    os.remove(f)
  for f in glob.glob(f'{output_path}/plot*.png'):
    os.remove(f)


# Initialize
c_emax = 0.973
c_emin = 0.635
c_ein = -Dr*(c_emax-c_emin)+c_emax # void ratio

dStrain = np.zeros((3,3))
'''
dStrain[0,0] = dstrain
dStrain[1,1] = -dstrain*0.5 # undrained
dStrain[2,2] = -dstrain*0.5
'''
dStrain[0,1] = dstrain
dStrain[1,0] = dstrain

ifDetailed = False
ifReversed = False
premaxtau = maxtau * 0.995

result = [[0,c_pin,0,0,0,0]] #[eps,p,tau]
#print('# , substeps, shear strain, p, tau')

ireverse = 0
istep = 0
iprint = 0

# calculate
cycliq_1ele = CycLiq(np.identity(3)*c_pin, c_ein)

'''
nstep = 13831
while istep<nstep:
'''
while ireverse<nreverse:
  try:
    cycliq_1ele.calc_mainstep(dStrain)
  except Exception as e:
    print('Error at istep', istep, ':', e, '\n')
    traceback.print_exc()
    exit(1)
  t_eps1 = 2*cycliq_1ele.Strn_now[0,1]
  t_p = cycliq_1ele.p_now
  t_Strs_dev = cycliq_1ele.Strs_dev_now.copy()
  #t_q = (1.5*np.tensordot(t_Strs_dev,t_Strs_dev))**0.5
  t_tau = cycliq_1ele.Strs_now[0,1]
  t_loadindex = cycliq_1ele.loadindex
  t_strn_vol_ir = cycliq_1ele.strn_vol_ir_now
  t_strn_vol_re = cycliq_1ele.strn_vol_re_now
  result.append([t_eps1,t_p,t_tau,t_loadindex,t_strn_vol_ir,t_strn_vol_re])
  if False:
  #if t_p < 2 and iprint < 15:
    with open(f'{output_path}/vars_{ireverse}_{istep}.txt', 'w') as f:
      pprint.pprint(vars(cycliq_1ele), stream=f)
    plot()
    iprint += 1
  #if t_q>maxq:
  #if abs(t_tau)>maxtau:
  #if t_tau>maxtau or t_tau<1.e-2:
  if t_tau>maxtau or t_tau<-8:
    if istep<5:
      continue
    dStrain *= -1
    ireverse += 1
    iprint = 0
    print('Loading reversed', ireverse, 'at istep', istep)
    with open(f'{output_path}/vars_{ireverse}_{istep}.txt', 'w') as f:
      pprint.pprint(vars(cycliq_1ele), stream=f)
    plot()
    '''
    '''
  '''
  if abs(t_tau)>premaxtau:
    if not ifDetailed and not ifReversed:
      dStrain *= 0.1
      ifDetailed = True
    elif abs(t_tau)>maxtau and not ifReversed:
      ifReversed = True
      ireverse += 1
      dStrain *= -10
      ifDetailed = False
      print('Loading reversed', ireverse, 'at istep', istep)
      with open(f'{output_path}/vars_{ireverse}_{istep}.txt', 'w') as f:
        pprint.pprint(vars(cycliq_1ele), stream=f)
      plot()
  else:
    if ifReversed:
      ifReversed = False
  '''
  istep += 1
print('nstep =', istep)
plot(True)


# visualization
