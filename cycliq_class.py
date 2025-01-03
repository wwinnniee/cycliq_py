import numpy as np
import math
import pprint

'''
In python, mutable vars are called by reference inside funcs.
'''

### Constants ### ------------------------------------------------------------

'''
naming rule
Namestartwithcap : basically vectors or matrices
c_name : constants
#g_name : global variables (not for class variables)
a_name : input as arguments
p_name : local variables in pegasus procedure
s_name : local variables in substep
t_name : temporal use
i_name : count variables (local)
'''

c_G0 = 200.
c_kappa = 0.008
c_h = 1.8
#c_M = 1.25 # Toyoura a
c_M = 1.35 # Toyoura b
#c_dre2 = 0.6 # Toyoura a
c_dre1 = 0.35 # Toyoura b
c_dre2 = 30.
#c_dir = 1.4 # Toyoura a
c_dir = 0.75 # Toyoura b
c_alpha = 20. # "eta"
c_gammadr = 0.05 # "rdr"
c_np = 1.1
c_nd = 7.8
c_lambdac = 0.019
c_e0 = 0.934
c_xi = 0.7

c_pat = 101.
c_pmin = 0.5
c_tolerance_pmin = 1e-8*c_pmin
c_tolerance_yield6 = 1e-6
c_tolerance_yield = 1e-5
c_tolerance_deta = 0.05 # to determin the number of substeps
c_tolerance_dgamma = 0.001 # to determin the number of substeps
c_tolerance_L = 1e-5

c_1_3 = 1./3.
c_I = np.identity(3)

c_max_iteration = 5e2


### CycLiq Class ### --------------------------------------------------------

class CycLiq:
  
  ### constructor ### ---------------------------- {
  def __init__(self, a_Strs_in, a_c_ein, \
               DEBUG=False, debug_vars=None):
    # Debug mode
    self.DEBUG = DEBUG
    if DEBUG:
      a_strn_vol_ir_pre = debug_vars[0]
      a_strn_vol_re_pre = debug_vars[1]
      a_strn_vol_c_0 = debug_vars[2]
      a_strn_vol_c_pre = debug_vars[3]
      a_r_alpha =  np.array([[debug_vars[4], debug_vars[5], debug_vars[6]], \
                             [debug_vars[7], debug_vars[8], debug_vars[9]], \
                             [debug_vars[10], debug_vars[11], debug_vars[12]]])
      a_M_max = debug_vars[13]
      a_strn_vol_ir_reversal = debug_vars[14]
      a_gammamono = debug_vars[15]
      a_strn_vol_pre = debug_vars[16]
      a_psi = debug_vars[17]

    # Updated only in the end of each main step and const during each step
    self.Strs_pre = a_Strs_in
    self.Strn_pre = np.zeros((3,3))
    self.Strs_now = None # np.zeros((3,3))
    self.Strn_now = None # np.zeros((3,3))

    # Updated and trialed in each substep
    self.p_pre = np.trace(a_Strs_in)/3.
    self.Strs_dev_pre = a_Strs_in-self.p_pre*c_I
    self.r_pre = (a_Strs_in-self.p_pre*c_I)/self.p_pre

    # Updated in the end of each substep and const during each step
    self.strn_vol_pre = a_strn_vol_pre if DEBUG else np.trace(self.Strn_pre)/3.
    self.Strn_dev_pre = self.Strn_pre-self.strn_vol_pre*c_I
    self.p_now = None
    self.Strs_dev_now = None # np.zeros((3,3))
    self.r_now = None # np.zeros((3,3))
    self.strn_vol_now = None # scalar, "trace_n"
    self.Strn_dev_now = None # np.zeros((3,3))
    self.psi = None # scalar
    self.gtheta = None # scalar, value of interpolation g when lode angle theta
    self.G = None # scalar
    self.K = None # scalar
    self.strn_vol_ir_pre = a_strn_vol_ir_pre if DEBUG else 0.
    self.strn_vol_re_pre = a_strn_vol_re_pre if DEBUG else 0.

    # Const during whole simulation
    self.c_ein = a_c_ein
    self.c_pin = self.p_pre
    #   supposed const
    self.M_peak = c_M
    t_sinphi = 3.*self.M_peak/(6.+self.M_peak)
    t_tanphi = t_sinphi/(1.-t_sinphi**2)**0.5
    self.M_peako = 2.*3.**0.5*t_tanphi/(3.+4.*t_tanphi**2)**0.5
    self.strn_vol_c_0 = a_strn_vol_c_0 if DEBUG else -2*c_kappa/(1+self.c_ein)*((self.c_pin/c_pat)**0.5-(c_pmin/c_pat)**0.5)

    # Updated with the update of p_pre 
    #   and can be calculated after the definition of the constants above
    self.set_psi()
    if DEBUG:
      self.psi = a_psi # difference because of p_pre and trace(sig_all-u)
    self.gtheta = self.get_gtheta(self.r_pre)
    self.set_GK(self.p_pre)

    # taken over steps and updated through simulation
    self.strn_vol_c_pre = a_strn_vol_c_pre if DEBUG else 0.
    self.r_alpha = a_r_alpha if DEBUG else self.r_pre # reversal point in deviatoric stress point, alpha
    self.M_max = a_M_max if DEBUG else (1.5*np.tensordot(self.r_pre,self.r_pre))**0.5/self.gtheta
    if self.M_max<c_tolerance_pmin:
      self.M_max = c_tolerance_pmin

    self.strn_vol_ir_reversal = a_strn_vol_ir_reversal if DEBUG else 0. # the value at last load reversal, used to calc. chi
    self.gammamono = a_gammamono if DEBUG else 0 # shear strain since the last load reversal, used to calc. irreversal dilatancy

    # Most variables are initialized in self.calc_init_classvar() 
    #  which is called in the beginning of every step.
    # Therefore, use of some funcs or variables before the first main step
    #  may cause errors.

    '''
    # Given initial value in each main step
    #   const but not initialized in self.calc_init_classvar()
    self.nsub = None # int, number of substeps

    # Used in each step
    self.strn_vol_c_now = None # scalar

    # Used inside each substep
    self.strn_vol_ir_now = None # scalar
    self.strn_vol_re_now = None # scalar
    self.r_bar = None # np.zeros((3,3)), previous load reversal point
    self.r_d = None # np.zeros((3,3)), reversible dilatancy surface
    self.dila_re = None # scalar, reversible dilatancy
    self.dila_ir = None # scalar, irreversible dilatancy
    self.chi = None # scalar, releasing reversible dilatancy
    self.dila_all = None # scalar, total dilatancy
    self.loadindex = None # scalar, loading index, L, "lambda", NOT "loadindex"
    self.plast_modul = None # scalar, plastic modulus, H, Kp
    self.Normal = None # np.zeros((3,3)), normal vector of yield surface
    self.r_dist_ratio = None # scalar, rho/rho_bar, "beta"
    self.phi = None # scalar, yield function
    #self.dStrn_dev_p = None # np.zeros((3,3)) # plastic deviatoric strain increment # s_dStrn_dev_p
    #self.dstrn_vol_p = None # scalar, plastic volumetric strain increment # s_dstrn_vol_p

    self.gammamono = None # scalar, init inside constructor?
    '''

    '''
    self.IbunI = np.zeros((3,3,3,3)) # K component of elast. tangent tensor
    self.IIdev = np.zeros((3,3,3,3)) # G component of elast. tangent tensor

    self.IbunI[0][0][0][0] = 1.0 
    self.IbunI[0][0][1][1] = 1.0 
    self.IbunI[0][0][2][2] = 1.0 
    self.IbunI[1][1][0][0] = 1.0 
    self.IbunI[1][1][1][1] = 1.0 
    self.IbunI[1][1][2][2] = 1.0 
    self.IbunI[2][2][0][0] = 1.0 
    self.IbunI[2][2][1][1] = 1.0 
    self.IbunI[2][2][2][2] = 1.0 

    self.IIdev[0][0][0][0] = 2./3.
    self.IIdev[0][0][1][1] = -1./3.
    self.IIdev[0][0][2][2] = -1./3.
    self.IIdev[0][1][0][1] = 0.5 
    self.IIdev[0][1][1][0] = 0.5 
    self.IIdev[0][2][0][2] = 0.5 
    self.IIdev[0][2][2][0] = 0.5 
    self.IIdev[1][0][0][1] = 0.5 
    self.IIdev[1][0][1][0] = 0.5 
    self.IIdev[1][1][0][0] = -1./3.
    self.IIdev[1][1][1][1] = 2./3.
    self.IIdev[1][1][2][2] = -1./3.
    self.IIdev[1][2][1][2] = 0.5 
    self.IIdev[1][2][2][1] = 0.5 
    self.IIdev[2][0][0][2] = 0.5 
    self.IIdev[2][0][2][0] = 0.5 
    self.IIdev[2][1][1][2] = 0.5 
    self.IIdev[2][1][2][1] = 0.5 
    self.IIdev[2][2][0][0] = -1./3.
    self.IIdev[2][2][1][1] = -1./3.
    self.IIdev[2][2][2][2] = 2./3.
    '''

    #make sure all the vars in self.calc_init_classvar
  ### } -------------------------------------------

  ### -----------------------------------------
  #def update_state(self):
  def update_state(self, a_dstrn_vol):
    self.strn_vol_pre = self.strn_vol_now # updated in each supstep
    self.Strn_dev_pre = self.Strn_dev_now # updated in each supstep
    '''
    self.strn_vol_pre = c_1_3*(self.Strn_pre[0,0]+self.Strn_pre[1,1]+self.Strn_pre[2,2])
    self.Strn_dev_pre = self.Strn_pre-self.strn_vol_pre*c_I
    '''
    self.strn_vol_ir_pre = self.strn_vol_ir_now
    self.strn_vol_re_pre = self.strn_vol_re_now
    #self.strn_vol_c_pre = self.strn_vol_c_now
    self.strn_vol_c_pre = self.strn_vol_c_pre-self.strn_vol_c_now+a_dstrn_vol-self.loadindex*self.dila_all
    self.strn_vol_c_now = self.strn_vol_c_pre
    self.p_pre = self.p_now # updated in each supstep
    self.Strs_dev_pre = self.Strs_dev_now
    self.r_pre = self.r_now # updated in each supstep
    self.set_psi()
    self.gtheta = self.get_gtheta(self.r_pre)
    self.update_M_max(self.r_pre,self.gtheta)
    self.set_GK(self.p_pre)

  ### -----------------------------------------
  def init_next_state(self):
    self.strn_vol_now = None
    self.Strn_dev_now = None
    self.strn_vol_ir_now = None
    self.strn_vol_re_now = None
    self.strn_vol_c_now = None
    self.p_now = None
    self.Strs_dev_now = None
    self.r_now = None

  ### -----------------------------------------
  def init_substep_vars(self):
    '''
    self.set_strn_vol_c_0()

    # lode angle
    self.M_peak = c_M*math.exp(-c_np*self.psi)
    t_sinphi = 3.*self.M_peak/(6.+self.M_peak)
    t_tanphi = t_sinphi/(1.-t_sinphi**2)**0.5
    self.M_peako = 2.*3.**0.5*t_tanphi/(3.+4.*t_tanphi**2)**0.5
    '''

    self.r_bar = None # np.zeros((3,3))
    self.r_d = None # np.zeros((3,3))
    self.dila_re = None # scalar, dilatancy
    self.dila_ir = None # scalar
    self.chi = None # scalar, releasing reversible dilatancy
    self.dila_all = None # scalar
    #self.dila_all = 0. # scalar
    self.loadindex = None # scalar, loading index, L, "lambda", NOT "loadindex"
    self.plast_modul = None # scalar, plastic modulus, H, Kp
    self.Normal = None # np.zeros((3,3))
    self.r_dist_ratio = None # scalar, rho/rho_bar, "beta"
    self.phi = None # scalar, yield function
    #self.dStrn_dev_p = None # np.zeros((3,3)) # plastic deviatoric strain increment
    #self.dstrn_vol_p = None # scalar, plastic volumetric strain increment

  ### -----------------------------------------
  def update_mainstep(self):
    self.Strn_pre = self.Strn_now.copy() # updated in the end of mainstep
    self.Strs_pre = self.Strs_now.copy() # updated in the end of mainstep
    #self.update_state()
    '''
    self.Strn_now = None
    self.Strs_now = None
    '''

  ### -----------------------------------------
  def set_mainstep_next_Strain(self, a_dStrain): # strain increment (dEpsilon)
    self.Strn_now = self.Strn_pre+a_dStrain
    self.strn_vol_now = c_1_3*(self.Strn_now[0,0]+self.Strn_now[1,1]+self.Strn_now[2,2])
    self.Strn_dev_now = self.Strn_now-self.strn_vol_now*c_I

  ### -----------------------------------------
  def set_substep_next_Strain(self, a_dstrn_vol, a_dStrn_dev): # vol. and dev. strain increment
    self.strn_vol_now = self.strn_vol_pre+a_dstrn_vol
    self.Strn_dev_now = self.Strn_dev_pre+a_dStrn_dev
    '''
    self.p_now = None
    self.r_now = None
    '''

  ### -----------------------------------------
  def set_p(self, a_elast_vol_strain):
    if a_elast_vol_strain>self.strn_vol_c_0:
      '''
      self.p_now = c_pat*((self.c_pin/c_pat)**0.5+(1+self.c_ein)*0.5/c_kappa*a_elast_vol_strain)**2
      '''

      self.p_now = c_pat*((self.p_pre/c_pat)**0.5+(1+self.c_ein)*0.5/c_kappa*a_elast_vol_strain)**2
    else:
      self.p_now = c_pmin
    #self.set_GK((self.p_pre+self.p_now)*0.5)

  ### -----------------------------------------
  def add_elast_increment(self, a_dstrn_vol_c, a_dStrn_dev):
    self.strn_vol_c_now = self.strn_vol_c_pre+a_dstrn_vol_c
    self.set_p(self.strn_vol_c_now)
    self.Strs_dev_now = self.Strs_dev_pre+2.*self.G*a_dStrn_dev
    self.r_now = self.Strs_dev_now/self.p_now

  ### -----------------------------------------
  def set_elast_state(self):
    #self.strn_vol_now = None
    #self.Strn_dev_now = None
    self.strn_vol_ir_now = self.strn_vol_ir_pre
    self.strn_vol_re_now = self.strn_vol_re_pre
    #self.strn_vol_c_now = None
    #self.p_now = None
    #self.Strs_dev_now = None
    #self.r_now = None
    self.dila_all = 0

  ### -----------------------------------------
  def set_psi(self):
    # critical state parameters
    t_e_pre = (1+self.c_ein)*math.exp(-self.strn_vol_pre) - 1. # en
    t_ec = c_e0-c_lambdac*(self.p_pre/c_pat)**c_xi 
    self.psi = t_e_pre-t_ec

  ### -----------------------------------------
  def set_GK(self, a_p): # mean effective stress (p)
    self.G = c_G0*(2.973-self.c_ein)**2/(1.+self.c_ein)*(c_pat*a_p)**0.5
    self.K = (1.+self.c_ein)/c_kappa*(c_pat*a_p)**0.5
    '''
    #print('   set_GK; G =',self.G,' K =',self.K)
    c_nu = 0.25
    t_K_with_nu = 2./3.*self.G*(1+c_nu)/(1-2*c_nu)
    #print('   K_wang/K_li :',self.K/t_K_with_nu)
    '''

  ### -----------------------------------------
  def update_M_max(self, a_current_strsratio, a_gtheta=1): # dev. stress ratio (r), g(theta)
    t_eta = (1.5*np.tensordot(a_current_strsratio,a_current_strsratio))**0.5
    t_M_now = t_eta/a_gtheta
    if t_M_now>self.M_max:
      self.M_max = t_M_now
    '''
    if t_M_now>self.M_peak-c_tolerance_pmin:
    '''
    if t_M_now>c_M*math.exp(-c_np*self.psi)-c_tolerance_pmin:
      self.M_max = t_M_now

  ### -----------------------------------------
  def get_gtheta(self, a_dev_strs_or_ratio): # both dev. stress (s) or stress ratio (r) are OK
    t_J2 = 0.5*np.tensordot(a_dev_strs_or_ratio,a_dev_strs_or_ratio)
    t_J3 = np.linalg.det(a_dev_strs_or_ratio)
    t_sin3theta = 0 if t_J2==0 else -0.5*t_J3*(3/t_J2)**1.5
    if t_sin3theta>1.:
      t_sin3theta = 1.
    elif t_sin3theta<-1.:
      t_sin3theta = -1.
    return 1/(1+self.M_peak*(t_sin3theta+t_sin3theta**2)/6+(self.M_peak/self.M_peako-1)*(1-t_sin3theta**2))
    #return 1.

  ### -----------------------------------------
  def get_yield_func(self, a_dev_strs_ratio): # dev. stress ratio (r)
    t_norm = np.tensordot(a_dev_strs_ratio,a_dev_strs_ratio)**0.5
    t_Normal = np.zeros((3,3)) if t_norm==0 else a_dev_strs_ratio/t_norm
    t_gtheta = self.get_gtheta(t_Normal)
    t_1 = (2/3.)**0.5*self.M_max*t_gtheta-t_norm
    t_2 = np.tensordot((2/3.)**0.5*self.M_max*t_gtheta*t_Normal-a_dev_strs_ratio,t_Normal).item()
    #return (2/3.)**0.5*self.M_max*t_gtheta-t_norm
    #print(np.tensordot(t_Normal,t_Normal),np.tensordot(a_dev_strs_ratio,t_Normal)-t_norm)
    #print(t_1, t_2)
    return t_2

  '''
  def set_strn_vol_c_0(self):
    self.strn_vol_c_0 = -2*c_kappa/(1+self.c_ein)*((self.p_pre/c_pat)**0.5-(c_pmin/c_pat)**0.5)
  '''

  ### -----------------------------------------
  def set_r_bar(self):
    '''
    if np.tensordot(self.r_pre,self.r_pre)<c_tolerance_pmin and \
       np.tensordot(self.r_alpha,self.r_alpha)<c_tolerance_pmin:
    '''
    if np.tensordot(self.r_pre,self.r_pre)<c_tolerance_pmin and \
       np.tensordot(self.r_alpha,self.r_alpha)<c_tolerance_pmin:
      '''
      self.Normal = 0.2**0.5*np.array([[2,0,0],[0,-1,0],[0,0,-1]])
      self.r_bar = (2/3.)**0.5*c_M*math.exp(-c_np*self.psi)*self.Normal
      '''
      self.r_bar = (2/15)**0.5*c_M*math.exp(-c_np*self.psi)*np.array([[2,0,0],[0,-1,0],[0,0,-1]])
      self.r_dist_ratio = 1.e20
      # return 0
    elif np.tensordot(self.r_pre-self.r_alpha,self.r_pre-self.r_alpha)<c_tolerance_pmin:
      '''
      self.Normal = self.r_pre/np.tensordot(self.r_pre,self.r_pre)
      self.r_bar = (2/3.)**0.5*self.M_max*self.Normal
      #self.r_bar = (2/3.)**0.5*self.M_max*sin3thetaself.Normal
      '''
      self.r_bar = (2/3.)**0.5*self.M_max*self.gtheta/np.tensordot(self.r_pre,self.r_pre)*self.r_pre
      self.r_dist_ratio = 1.e20
      # return 0
    else:
    ### Pegasus procedure

      ### PP 1. Initialization
      p_beta0 = 0. if np.tensordot(self.r_alpha,self.r_alpha)>c_tolerance_pmin else 0.01
      p_beta1 = 1.
      p_rbar0 = (1-p_beta0)*self.r_alpha+p_beta0*self.r_pre
      p_rbar1 = (1-p_beta1)*self.r_alpha+p_beta1*self.r_pre
      p_beta = None
      p_rbar = None
      p_Fm = None

      ### PP 2. 
      p_Fm0 = self.get_yield_func(p_rbar0) # value of yielding function (max stress ratio surface) at rbar0
      p_Fm1 = self.get_yield_func(p_rbar1)

      ### PP 3. 
      if abs(p_Fm1)<c_tolerance_yield:
        self.r_bar = p_rbar1.copy()
        #self.Normal = self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5
        self.r_dist_ratio = p_beta1
        # return 0
      elif abs(p_Fm0)<c_tolerance_yield:
        self.r_bar = p_rbar0.copy()
        #self.Normal = self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5
        self.r_dist_ratio = p_beta0
        # return 0
      else:
        i_while_PP1 = 0
        while p_Fm0*p_Fm1>0:

      ### PP 2. 
          p_beta0 = p_beta1
          p_beta1 = 2*p_beta1
          p_rbar0 = (1-p_beta0)*self.r_alpha+p_beta0*self.r_pre
          p_rbar1 = (1-p_beta1)*self.r_alpha+p_beta1*self.r_pre

          p_Fm0 = self.get_yield_func(p_rbar0)
          p_Fm1 = self.get_yield_func(p_rbar1)

          i_while_PP1 += 1
          if i_while_PP1 > c_max_iteration:
            raise Exception("while_PP1")

      ### PP 4. 
        if abs(p_Fm1)<c_tolerance_yield:
          self.r_bar = p_rbar1.copy()
          #self.Normal = self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5
          self.r_dist_ratio = p_beta1
          # return 0
        elif abs(p_Fm0)<c_tolerance_yield:
          self.r_bar = p_rbar0.copy()
          #self.Normal = self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5
          self.r_dist_ratio = p_beta0
          # return 0
        else:
          p_beta = p_beta1-p_Fm1*(p_beta1-p_beta0)/(p_Fm1-p_Fm0)
          p_rbar = (1-p_beta)*self.r_alpha+p_beta*self.r_pre
          p_Fm = self.get_yield_func(p_rbar)

      ### PP 5. 
          i_while_PP2 = 0
          while abs(p_Fm) > c_tolerance_yield6:
            if p_Fm*p_Fm1<0:
              p_beta0 = p_beta1
              p_Fm0 = p_Fm1
              p_beta1 = p_beta
              p_Fm1 = p_Fm
            else:
              # p_Fm0 = p_Fm0*p_Fm1/(p_Fm+p_Fm1)
              p_Fm0 = p_Fm0*p_Fm1/(p_Fm0+p_Fm1)
              p_beta1 = p_beta
              p_Fm1 = p_Fm

      ### PP 4. 
            p_beta = p_beta1-p_Fm1*(p_beta1-p_beta0)/(p_Fm1-p_Fm0)
            p_rbar = (1-p_beta)*self.r_alpha+p_beta*self.r_pre
            p_Fm = self.get_yield_func(p_rbar)

            i_while_PP2 += 1
            if i_while_PP2 > c_max_iteration:
              raise Exception("while_PP2")

          self.r_bar = p_rbar.copy()
          #self.Normal = self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5
          self.r_dist_ratio = p_beta

  ### -----------------------------------------
  def set_dilatancy(self):
    self.r_d = c_M*math.exp(c_nd*self.psi)/self.M_max*self.r_bar
    self.dila_re = (2./3.)**0.5*c_dre1*np.tensordot(self.r_d-self.r_pre,self.Normal)
    if self.strn_vol_ir_reversal>c_tolerance_pmin:
      #self.chi = -c_dir*self.strn_vol_re_pre/self.strn_vol_ir_pre
      self.chi = -c_dir*self.strn_vol_re_pre/self.strn_vol_ir_reversal
    else:
      self.chi = 0
    '''
    strn_vol_ir_pre's'?
    '''
    if self.chi>1.:
      self.chi = 1.
    self.dila_ir = 0
    '''
    if self.dila_re>0:
      #self.dila_re = (c_dre2*self.chi)**2/self.p_pre
      self.dila_re = (c_dre2*self.chi)**2/max(self.p_pre,1.)
    '''
    t1 = c_dir*math.exp(c_nd*self.psi-c_alpha*self.strn_vol_ir_pre)
    t2 = (2/3.)**0.5*np.tensordot(self.r_d-self.r_pre,self.Normal)*math.exp(self.chi)
    t3 = (c_gammadr*(1-math.exp(c_nd*self.psi))/(c_gammadr*(1-math.exp(c_nd*self.psi))+self.gammamono))**2
    if self.dila_re>0:
      if self.psi>0:
        self.dila_ir = t1*t2
      else:
        self.dila_ir = t1*(t2+t3)
      '''
      if -epsvre_ns<tolerance: dre_n=0
      '''
    else:
      if self.psi>0:
        self.dila_ir = 0
      else:
        self.dila_ir = t1*t3
    '''
    '''
    if self.dila_re>0:
      #self.dila_re = (c_dre2*self.chi)**2/self.p_pre
      self.dila_re = (c_dre2*self.chi)**2/max(self.p_pre,1.)
    self.dila_all = self.dila_re+self.dila_ir

  ### -----------------------------------------
  def calc_substep_CP(self, a_dstrn_vol, a_dStrn_dev):
    ### sub 1. Initialization ### --------------------------------------------------
    self.init_next_state()
    self.set_substep_next_Strain(a_dstrn_vol, a_dStrn_dev)
    self.init_substep_vars()

    '''
    self.set_strn_vol_c_0()
    '''
    s_dstrn_vol_p = 0
    s_dStrn_dev_p = np.zeros((3,3)) # plastic deviatoric strain increment
    s_rNormal = 0
    self.loadindex = 0

    ### sub 2. Elastic prediction
    self.add_elast_increment(a_dstrn_vol, a_dStrn_dev)
    #self.update_M_max(self.r_now,self.gtheta)
    self.set_r_bar()
    #self.Normal *= 1.5**0.5
    self.Normal = 1.5**0.5*self.r_bar/np.tensordot(self.r_bar,self.r_bar)**0.5

    ### sub 3. Consistency condition
    s_rNormal = np.tensordot(self.r_pre,self.Normal)
    self.phi = np.tensordot(self.Strs_dev_now-self.Strs_dev_pre,self.Normal)-(self.p_now-self.p_pre)*s_rNormal
    t_phi_n = np.tensordot(self.r_now-self.r_pre,self.Normal)
    # load reversal
    if self.phi<c_tolerance_pmin or t_phi_n<c_tolerance_pmin:
      self.gammamono = 0.
      self.r_alpha = self.r_pre
      self.strn_vol_ir_reversal = self.strn_vol_ir_pre
      '''
      self.strn_vol_ir_pre = self.strn_vol_ir_pre
      '''
      self.set_elast_state()
      #print('Alpha relocated :', self.r_alpha)

    else:
    ### sub 4. Plastic correction
      ifLMinus= False
      self.loadindex = 0.
      '''
      self.strn_vol_c_now = 
      '''
      if (1.5*np.tensordot(self.r_bar-self.r_alpha,self.r_bar-self.r_alpha))**0.5>c_tolerance_pmin:
        self.plast_modul = 2/3.*c_h*self.gtheta*self.G*math.exp(-c_np*self.psi)* \
                            (c_M*math.exp(-c_np*self.psi)/self.M_max*self.r_dist_ratio-1.)
        if self.plast_modul<c_tolerance_pmin and self.plast_modul>=0:
          self.plast_modul = c_tolerance_pmin
        elif self.plast_modul>-c_tolerance_pmin and self.plast_modul<0:
          self.plast_modul = -c_tolerance_pmin
        self.set_dilatancy()
        t_iconv = False
        t_loadindex_max = 0
        t_loadindex_min = 0
        i_while_CP = 0
        while True: # do
          if not t_iconv:
            self.loadindex += self.phi/(self.plast_modul+2*self.G-self.K*self.dila_all*np.tensordot(self.r_now,self.Normal))
            if self.loadindex<0:
              self.loadindex = c_tolerance_L
              ifLMinus= True
          else:
            self.loadindex = 0.5*(t_loadindex_max+t_loadindex_min)
          s_dstrn_vol_p = self.loadindex*self.dila_all
          s_dStrn_dev_p = self.loadindex*self.Normal
          self.strn_vol_c_now = self.strn_vol_c_pre+a_dstrn_vol-s_dstrn_vol_p
          if self.strn_vol_c_now<=self.strn_vol_c_0:
            self.strn_vol_c_now = self.strn_vol_c_0
            '''
          self.set_p(self.strn_vol_c_now)
            '''
            self.p_now = c_pmin
          else:
            self.set_p(self.strn_vol_c_now)
          #self.set_GK()
          self.Strs_dev_now = self.Strs_dev_pre+2.*self.G*(a_dStrn_dev-s_dStrn_dev_p)

    ### sub 5. Convergence of consistency condition
          self.phi = np.tensordot(self.Strs_dev_now-self.Strs_dev_pre,self.Normal)-\
                      (self.p_now-self.p_pre)*s_rNormal-self.loadindex*self.plast_modul

          if self.phi<-c_tolerance_pmin:
            t_iconv = True
            t_loadindex_max = self.loadindex
          elif self.phi>c_tolerance_pmin and t_iconv:
            t_loadindex_min = self.loadindex
          self.strn_vol_ir_now = self.strn_vol_ir_pre+self.loadindex*self.dila_ir
          self.strn_vol_re_now = self.strn_vol_re_pre+self.loadindex*self.dila_re

          if abs(self.phi)<c_tolerance_pmin: # while
            break

          i_while_CP += 1
          if self.DEBUG:
            with open(f'./output_debug/vars_{i_while_CP}.txt', 'w') as f:
              pprint.pprint(vars(self), stream=f)
            with open('./output_debug/ld-phi.txt', 'a') as f:
              f.write(f'{self.loadindex}, {self.phi}\n')
          if i_while_CP > c_max_iteration:
            raise Exception("while_CP")
        # End while
        self.gammamono += self.loadindex
        if ifLMinus:
          #pprint.pprint(vars(self))
          pass

  ### sub 6. Update variables
      '''
      932-952 r_nplus1 update?
      '''
      self.r_now = self.Strs_dev_now/self.p_now
    self.update_state(a_dstrn_vol)

  ### -----------------------------------------
  def init_mainstep(self):
    self.Strs_now = None
    self.nsub = None # scalar

  ### -----------------------------------------
  def calc_mainstep(self, a_dStrn):
    self.set_mainstep_next_Strain(a_dStrn)
    self.init_mainstep()
    self.strn_vol_c_0 = -2*c_kappa/(1+self.c_ein)*((self.p_pre/c_pat)**0.5-(c_pmin/c_pat)**0.5)
    #if abs(self.strn_vol_c_0)<c_tolerance_pmin:
    #  self.strn_vol_c_0 = 0

    ### trial before substep
    self.add_elast_increment(self.strn_vol_now-self.strn_vol_pre, self.Strn_dev_now-self.Strn_dev_pre)

    t_Increment = self.r_now-self.r_pre
    t_nsub1 = math.ceil((1.5*np.tensordot(t_Increment,t_Increment))**0.5/c_tolerance_deta)
    t_Increment = self.Strn_dev_now-self.Strn_dev_pre
    t_nsub2 = math.ceil((2/3*np.tensordot(t_Increment,t_Increment))**0.5/c_tolerance_dgamma)
    self.nsub = min(100, max(t_nsub1, t_nsub2))

    ### substep
    t_dstrn_vol = (self.strn_vol_now-self.strn_vol_pre)/self.nsub
    t_dStrn_dev = (self.Strn_dev_now-self.Strn_dev_pre)/self.nsub
    for i_sub in range(self.nsub):
      self.calc_substep_CP(t_dstrn_vol, t_dStrn_dev)
    ### End substeps

    # Continuum tangent operator (52)
    '''
    self.set_GK(0.5*(self.p_now+self.p_pre))
    '''

    '''
    self.Tangent_e = self.K*self.IbunI+2*self.G*self.IIdev
    '''

    '''
    self.psi?
    self.r_pre = self.r_now?
    t_rn =
    self.set_r_bar()
    self.Normal = 
    self.set_plast_modul(psi,r_dist
    self.set_dilatancy(psi,r
    '''

    '''
    t_R = self.Normal+1/3*self.dila_all*c_I
    t_L = self.Normal-1/3*t_rn*c_I
    t_DRLD = np.einsum("ij,kl->ijkl",np.einsum("ijkl,kl->ij",self.Tangent_e,t_R),np.einsum("ij,ijkl->kl",t_L,self.Tangent_e)
    t_LDR = np.tensordot(np.einsum("ij,ijkl->kl",t_L,self.Tangent_e),t_R)
    self.Tangent_ep = self.Tangent_e-t_DRLD/(self.plast_modul+t_LDR)
    '''

    # Update stress
    self.Strs_now = self.Strs_dev_now + self.p_now*c_I
    self.update_mainstep()

