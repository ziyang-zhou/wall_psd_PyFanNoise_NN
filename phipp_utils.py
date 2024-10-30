""" Module that define phipp models"""

def calc_phipp(istrip,flow_params,omega,model='Willmarth-Roos-Amiet'):

    if model == 'Willmarth-Roos-Amiet':
        phipp=_phipp_WRA(istrip,flow_params,omega)
    elif model == 'Chase-Howe':
        phipp=_phipp_CH(istrip,flow_params,omega)
    elif model =='Goody':
        phipp=_phipp_Gy(istrip,flow_params,omega)
    elif model=='Kim-George':
        phipp=_phipp_KG(istrip,flow_params,omega)
    elif model=='Rozenberg':
        phipp=_phipp_RZ(istrip,flow_params,omega)
    elif model=='Gliebe':
        phipp=_phipp_Gl(istrip,flow_params,omega)
    elif model=='GEP2022':
        phipp=_gep_2_mod(istrip,flow_params,omega)
    elif model=='GEP2023':
        phipp=_gepnew_1(istrip,flow_params,omega)  
    elif model=='Lee':
        phipp=_phipp_LEE(istrip,flow_params,omega)
    elif model=='Pargal':
        phipp=_phipp_Pargal(istrip,flow_params,omega)
    elif model=='ANN':
        phipp=_phipp_ANN(istrip,flow_params,omega)
    else:
        raise IOError("Wrong model for phipp")
    return phipp

def _flow_param(istrip,flow_params):

    if 'U_ext' in flow_params.keys():
        key =  'U_ext'
    elif 'relative' in flow_params.keys():
        key = 'relative'
    elif 'absolute' in flow_params.keys():
        key = 'absolute'
    U_X = flow_params[key][istrip]

    if 'rho_ext' in flow_params.keys():
        key = 'rho_ext'
        rho = flow_params[key][istrip]
    elif 'rho' in flow_params.keys():
        key = 'rho'
        rho = flow_params[key][istrip]
    else:
        rho = 1.23 # TBC as input 

    q=0.5*rho*U_X**2
    return U_X,rho,q

def _compute_strouhal(U_X,delta,omega):
  
  strouhal_number=omega*delta/U_X

  return strouhal_number

def _Goody_param(istrip,flow_params,kinematic_viscosity=1.56e-5):
  from numpy import sqrt
  U_X,rho,q=_flow_param(istrip,flow_params)
  tau_wall = flow_params['tau_wall'][istrip]
  u_tau = sqrt(tau_wall/rho)
  Cf=tau_wall/q   # skin friction
  delta99 = flow_params['delta99'][istrip]
  
  RT=(u_tau*delta99/kinematic_viscosity)*sqrt(0.5*Cf) # ratio of time scales 

  return u_tau,Cf,RT  

def _phipp_WRA(istrip,flow_params,omega):
  
  U_X,rho,q=_flow_param(istrip,flow_params)

  delta_star=flow_params['delta_star'][istrip]
  strouhal_number=_compute_strouhal(U_X,delta_star,omega)
  ampl=delta_star*q**2/U_X*2.0e-5 
  
  phipp = ampl /(1.+strouhal_number+0.217*strouhal_number**2+0.00562*strouhal_number**4)
  return phipp

def _phipp_CH(istrip,flow_params,omega):
  
  U_X,rho,q=_flow_param(istrip,flow_params)

  delta_star=flow_params['delta_star'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]

  strouhal_number=_compute_strouhal(U_X,delta_star,omega)
  
  ampl=delta_star/U_X*tau_wall**2
  
  phipp =  ampl*strouhal_number**2 /((strouhal_number**2+0.0144)**1.5)
  return phipp

def _phipp_Gy(istrip,flow_params,omega):
  
  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]
  u_tau,Cf,RT=_Goody_param(istrip,flow_params)
  strouhal_number=_compute_strouhal(U_X,delta99,omega)
  C1=0.5
  C2=3.0
  C3=1.1*RT**(-0.57)

  # 0.5 comes from pos + negative frequency used in the published formulations
  ampl=0.5*tau_wall**2*delta99/U_X*C2
  
  phipp = ampl*strouhal_number**2 /((strouhal_number**0.75+C1)**3.7+(C3*strouhal_number)**7)
  return phipp

def _phipp_ANN(istrip,flow_params,omega):
  # Load the ANN
  from ANN_class import ANN_WPS_loader
  import AE_ANN_module
  import pandas as pd
  from scipy.interpolate import interp1d
  import numpy as np

  print('loading parameters for the ANN')
  ANN_loader = ANN_WPS_loader('/home/ziyz1701/storage/CD_airfoil/FAN_2025/NN_simplified/')
  ANN_loader.load_ANN()

  # Compute and load flow parameter

  U_X,_,_=_flow_param(istrip,flow_params)
  #Ut=flow_params['Ut'][istrip]#to add
  #h=flow_params['h'][istrip]#to add
  delta99=flow_params['delta99'][istrip]
  #Mach=flow_params['Mach']#to add
  #Re=flow_params['Re']#to add
  #alpha=flow_params['alpha']#to add
  #x_over_c=flow_params['x_over_c']#to add

  ######################Temporary input of dummy variables for testing############################
  St_interval = [0,250,255] # declare the strouhal numbers for the WPS prediction
  St = np.geomspace(St_interval[0]+1,St_interval[1]+1,St_interval[2])-1 #Strouhal number used as input for the ANN
  BL_df = pd.read_csv(ANN_loader.read_database_folder+'{:s}_BL_100.csv'.format(ANN_loader.testcase),header='infer',delimiter=' ')
  Ut = np.array(BL_df['Ut'])
  h = np.array(BL_df['h'])
  Mach = pd.Series([U_X/330.0], name='Mach')
  Re = pd.Series([150000/16.0*U_X], name='Re')
  alpha = pd.Series([8], name='alpha')
  Uinf = pd.Series([U_X], name='Uinf')
  x_over_c = 0.85 #fraction of chord of position from LE
  ################################################################################################

  #Compute the phipp using the ANN model
  phipp = AE_ANN_module.WPS_predictor_single(St,Ut,h,Mach,Re,alpha,Uinf,x_over_c,ANN_loader.model_WPS,ANN_loader.model_AE,ANN_loader.means,ANN_loader.stds)
  
  #Interpolate the result to obtain phipp at the desired St
  St_output=_compute_strouhal(U_X,delta99,omega) #Strouhal number used for interpolating output  
  interpolation_function  = interp1d(St, phipp, kind='linear', fill_value="extrapolate") #Create interpolation function
  phipp_output = interpolation_function(St_output) #Interpolate for the desired output St

  return phipp_output

def _phipp_KG(istrip,flow_params,omega,low_strouhal_condition=0.06):
  from numpy import zeros_like

  U_X,rho,q=_flow_param(istrip,flow_params)
  delta_star=flow_params['delta_star'][istrip]
  strouhal_number=_compute_strouhal(U_X,delta_star,omega)

  low_indices = (strouhal_number<low_strouhal_condition)
  high_indices=~low_indices
  low_St=strouhal_number[low_indices]
  high_St=strouhal_number[high_indices]

  phipp=zeros_like(strouhal_number)

  # 0.5 comes from pos + negative frequency used in the published formulations
  ampl=0.5 * delta_star*q**2/U_X 
  
  phipp[low_indices]= ampl * (1.732e-3 * low_St ) /(1.-5.439*low_St +36.74*low_St**2+0.1505*low_St**5)
  phipp[high_indices] = ampl * ( 1.4216e-3 * high_St ) /( 0.3261+4.1837*high_St+22.818*high_St**2 
      +0.0013*high_St**3 + 0.0028*high_St**5 )
  return phipp

def _phipp_RZ(istrip,flow_params,omega):
  from numpy import isnan,where,isinf

  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  delta_star=flow_params['delta_star'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]
  tau_max=tau_wall

  u_tau,Cf,RT=_Goody_param(istrip,flow_params)

  # option_explicit={'bl_thickness','displacement_thickness','tau_wall','momentum_thickness','pressure_gradient'};
  # option_implicit={'bl_thickness','displacement_thickness','tau_wall','PI_param','clauser_param'};
  # option implicit use of scipy.optimize.broyden1 tobe done
  # explicit for now
  pressure_gradient=max(flow_params['dpds'][istrip],0.)
  theta=flow_params['theta'][istrip]

  beta_c = (theta/tau_wall)*pressure_gradient #Clauser' parameter
  PI = 0.8*(beta_c+0.5)**0.75 #Law of the wake parameter (Durbi & Reif empirical formula)

  Delta_ratio = delta99/delta_star
  A1 = 3.7+1.5*beta_c
  A2 = 7 #Recover Goody's -5 slope
  F1=4.76*(1.4/Delta_ratio)**0.75*(0.375*A1-1)
  F2b=4.2*(PI/Delta_ratio)+1
  C3p=1.1*Delta_ratio*RT**(-0.57)
  F1b=(2.82*Delta_ratio**2 * (6.13*Delta_ratio**(-0.75) + F1)**A1 )

  strouhal_number=_compute_strouhal(U_X,delta_star,omega)  

  ampl=0.5 * tau_max**2*delta_star/U_X
  FFb=F1b*F2b
  # 0.5 comes from pos + negative frequency used in the published reference
  phipp = ampl * (FFb * strouhal_number**2) /( (4.76*strouhal_number**0.75+F1)**A1 + (C3p * strouhal_number)**A2 )
  if (isinf(phipp)).any():
    print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
    print ('F1b: ',F1b[where(isinf(phipp))[0][0],where(isinf(phipp))[1][0]])
    raise RuntimeError("Error in Rozenberg model")
    
  if (isnan(phipp)).any():
    print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
    raise RuntimeError("Error in Rozenberg model")
    # print F1[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print A1[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print C3p[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print Delta_ratio[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print delta99[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print delta_star[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
    # print RT[where(isnan(phipp))[0][0],where(isnan(phipp))[1][0]]
  return phipp

def _phipp_LEE(istrip,flow_params,omega):
    from numpy import isnan,where,isinf
    import numpy as np
    U_X,rho,q=_flow_param(istrip,flow_params)

    delta99=flow_params['delta99'][istrip]
    delta_star=flow_params['delta_star'][istrip]
    tau_wall=flow_params['tau_wall'][istrip]

    u_tau,Cf,RT=_Goody_param(istrip,flow_params)

    pressure_gradient=max(flow_params['dpds'][istrip],0.)
    theta=flow_params['theta'][istrip]    
    
    beta_c = (theta/tau_wall)*pressure_gradient #Clauser' parameter
    PI = 0.8*(beta_c+0.5)**0.75 #Law of the wake parameter (Durbi & Reif empirical formula)
    Delta_ratio = delta99/delta_star
    
    e=3.7+1.5*beta_c
    d=4.76*(1.4/Delta_ratio)**0.75*(0.375*e-1)
    a=2.82*Delta_ratio**2*(6.13*Delta_ratio**(-0.75)+d)**e * (4.2*(PI/Delta_ratio)+1)

    h_star=min(3,(0.139+3.1043*beta_c))+7
    if beta_c<0.5:
        d_star=max(1,1.5*d)
    else:
        d_star=d

    
    strouhal_number=_compute_strouhal(U_X,delta_star,omega)  

    ampl=0.5 * tau_wall**2*delta_star/U_X
    
    phipp= ampl * (max(a,(0.25*beta_c-0.52)*a)*(strouhal_number)**2)/((4.76*(strouhal_number)**0.75+d_star)**e+((8.8*RT**(-0.57))*(strouhal_number))**h_star)

    return phipp

def _phipp_Gl(istrip,flow_params,omega):

  U_X,rho,q=_flow_param(istrip,flow_params)

  delta_star=flow_params['delta_star'][istrip]
  
  strouhal_number=_compute_strouhal(U_X,delta_star,omega)

  ampl= 0.5 * rho**2 *delta_star*U_X**3 * 1.0e-4 
  # 0.5 for pos + neg fequencies

  phipp = ampl /(1.+0.5*(strouhal_number)**2)**(5./2.)

  return phipp



def _gep_1(istrip,flow_params,omega):
  from numpy import isnan,where,isinf
  import numpy as np
  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  delta_star=flow_params['delta_star'][istrip]

  u_tau,Cf,RT=_Goody_param(istrip,flow_params)

  pressure_gradient=max(flow_params['dpds'][istrip],0.)
  theta=flow_params['theta'][istrip]

  Delta_ratio = delta99/delta_star

  dels_Ue = delta_star/U_X
  Q2 = (0.5*rho*U_X**2)**2
  H = delta_star/theta
  G = np.sqrt(2.0/Cf) - np.sqrt(2.0/Cf)/H
  beta_c = delta99*np.abs(pressure_gradient/(0.5*rho*U_X**2))
  fgd = omega/2/np.pi
  Rt = RT
  
  SS = Q2*dels_Ue     # spectrum scaling
  FS = dels_Ue*fgd    # frequency scaling

  C1 = (SS)*(0.53367*G/H)**(Delta_ratio/H)*(FS)**2.98
  C2 = (SS)*((0.59*Delta_ratio)**(Delta_ratio*G))**(0.5*beta_c)*(FS)**1.066
  C3 = 211.55*8.0**(0.79*H)*(Rt**0.467*FS)**(2.575+0.79*H)
  C4 = 39322.57*(-0.266*Cf*Delta_ratio + 0.266*beta_c + (FS)**1.259 + 0.48)**8.0
  
  # division by 4pi for double sided wavenumber spectra
  phipp = (C1+C2)/(C3+C4)/(4*np.pi)


  if (isinf(phipp)).any():
    print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
    raise RuntimeError("Error in GEP-1 model")
    
  if (isnan(phipp)).any():
    print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
    raise RuntimeError("Error in GEP-1 model")
  return phipp

def _gep_2(istrip,flow_params,omega):
  from numpy import isnan,where,isinf
  import numpy as np

  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  delta_star=flow_params['delta_star'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]

  u_tau,Cf,RT=_Goody_param(istrip,flow_params)

  pressure_gradient=max(flow_params['dpds'][istrip],0.)
  theta=flow_params['theta'][istrip]

  Delta_ratio = delta99/delta_star

  
  tht_Ue = theta/U_X
  H = delta_star/theta
  beta_c = delta99*np.abs(pressure_gradient/(0.5*rho*U_X**2))
  fgd = omega/2/np.pi         # convert to frequency
  Rt = RT
  SS = tau_wall**2*tht_Ue     # spectrum scaling
  FS = fgd*tht_Ue             # frequency scaling

  C1 = (SS)*(-1.0*H*beta_c + 0.803*H + 2.744*beta_c - 2.205)*(FS)**1.113
  C2 = (SS)*(0.562*Delta_ratio**0.4047*(FS)**1.1226/(0.548**(1.0*beta_c)*Delta_ratio**(1.0*beta_c)))
  C3 = 26.245*(0.0941*(FS)**1.695902 + 0.000185)**(2.0842*H)/((FS)**1.6959 + 0.001965)**1.3827
  C4 = 85.294*4.402**(0.879*Delta_ratio)*4.402**H*(FS/Rt**0.457)**(3.0+0.879*Delta_ratio+H)
  
  # division by 4pi for double sided wavenumber spectra
  phipp = (C1+C2)/(C3+C4)/(4*np.pi)

  if (isinf(phipp)).any():
    print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
    raise RuntimeError("Error in GEP-2 model")
    
  if (isnan(phipp)).any():
    print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
    raise RuntimeError("Error in GEP-2 model")
  return phipp

def _gep_3(istrip,flow_params,omega):
  from numpy import isnan,where,isinf
  import numpy as np

  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  delta_star=flow_params['delta_star'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]

  u_tau,Cf,RT=_Goody_param(istrip,flow_params)

  pressure_gradient=max(flow_params['dpds'][istrip],0.)
  theta=flow_params['theta'][istrip]

  Delta_ratio = delta99/delta_star

  tht_Ue = theta/U_X
  H = delta_star/theta
  G = np.sqrt(2.0/Cf) - np.sqrt(2.0/Cf)/H
  beta_c = delta99*np.abs(pressure_gradient/(0.5*rho*U_X**2))
  fgd = omega/2/np.pi
  Rt = RT
  
  SS = tau_wall**2*tht_Ue   # spectrum scaling
  FS = fgd*tht_Ue    # frequency scaling 

  C1 = (SS)*(-1.29*Cf**2 + 2.58*Cf)*(FS)**1.306
  C2 = (SS)*(0.177*(Delta_ratio*beta_c)**3.3935*(FS)**0.872)
  C3 = (Cf/(H*(H + 2.0)) + 0.0009*(FS)**5.66)**1.944
  C4 = (0.293*FS/Rt**0.31)**(1.95*0.864**(G**beta_c)*H)
  
  # division by 4pi for double sided wavenumber spectra
  phipp = (C1+C2)/(C3+C4)/(4*np.pi)


  if (isinf(phipp)).any():
    print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
    raise RuntimeError("Error in GEP-3 model")
    
  if (isnan(phipp)).any():
    print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
    raise RuntimeError("Error in GEP-3 model")
  return phipp

def _gep_2_mod(istrip,flow_params,omega):
  from numpy import isnan,where,isinf
  import numpy as np

  U_X,rho,q=_flow_param(istrip,flow_params)

  delta99=flow_params['delta99'][istrip]
  delta_star=flow_params['delta_star'][istrip]
  tau_wall=flow_params['tau_wall'][istrip]

  u_tau,Cf,RT=_Goody_param(istrip,flow_params)

  pressure_gradient=max(flow_params['dpds'][istrip],0.)
  theta=flow_params['theta'][istrip]

  Delta_ratio = delta99/delta_star

  
  tht_Ue = theta/U_X
  H = delta_star/theta
  beta_c = delta99*np.abs(pressure_gradient/(0.5*rho*U_X**2))
  fgd = omega/2/np.pi         # convert to frequency
  Rt = RT
  SS = tau_wall**2*tht_Ue     # spectrum scaling
  FS = fgd*tht_Ue             # frequency scaling

  C1 = (SS)*(0.8734*Delta_ratio - 1.4894)*(FS)**1.2548
  C2 = (SS)*1.1950*beta_c*(FS)**1.2548
  C3 = 1.0544*(0.5590*Cf + (FS)**1.7888)**0.0911*(Cf + 1.7888*(FS)**1.7888)**(0.8927*H)
  C4 = 236570.2517*(FS/Rt**0.322)**11.25
  
  # division by 4pi for double sided wavenumber spectra
  phipp = (C1+C2)/(C3+C4)/(4*np.pi)

  if (isinf(phipp)).any():
    print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
    raise RuntimeError("Error in GEP-2-mod model")
    
  if (isnan(phipp)).any():
    print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
    raise RuntimeError("Error in GEP-2-mod model")
  return phipp

def _gepnew_1(istrip,flow_params,omega):
    from numpy import isnan,where,isinf
    import numpy as np
    U_X,rho,q=_flow_param(istrip,flow_params)

    delta99=flow_params['delta99'][istrip]
    delta_star=flow_params['delta_star'][istrip]
    tau_wall=flow_params['tau_wall'][istrip]

    u_tau,Cf,RT=_Goody_param(istrip,flow_params)

    pressure_gradient=max(flow_params['dpds'][istrip],0.)
    theta=flow_params['theta'][istrip]    
    
    Delta_ratio = delta99/delta_star
    
    
    tht_Ue = theta/U_X
    dels_Ue = delta_star/U_X
    del_Ue = delta99/U_X

    H = delta_star/theta
    beta_c4 = delta99*np.abs(pressure_gradient/(0.5*rho*U_X**2))
    fgd = omega/2/np.pi         # convert to frequency
    Rt = RT
    SS = tau_wall**2*tht_Ue     # spectrum scaling
    FS = fgd*tht_Ue             # frequency scaling
    G = np.sqrt(2.0/Cf) - np.sqrt(2.0/Cf)/H
    delt = delta99/delta_star
    tau_w2 = tau_wall**2
    Q2 = (0.5*rho*U_X**2)**2
    
    C1 = Cf*Q2*tht_Ue*(del_Ue*fgd)**1.296381
    C2 = Q2*tht_Ue*(del_Ue*fgd)**0.78206*(Cf*G + beta_c4 - 0.019653)**2.0
    C3 = 4.039168**((1.272276**H)**0.429996*(1.272276**H)**H)*(Rt**0.429996*fgd*tht_Ue)**((1.272276**H)**0.429996*(1.272276**H)**H)
    C4 = (2.151447*(del_Ue*fgd)**1.850338 + 0.747821)**H*(2.151447*(del_Ue*fgd)**1.850338 + 0.747821)**(4.039168/delt**1.0)/(2.151447*(del_Ue*fgd)**1.850338 + 0.747821)**(0.033104*G)
    # C5 = del_Ue*tau_w2*(G**0.133311*(del_Ue*fgd)**2.4923 + 4.864402*G*H*(dels_Ue*fgd)**0.789109)*((1.661557*Rt**0.021037*fgd*tht_Ue)**(2.4923*Cf + H + 6.27992) + (0.00335**(2.661557*beta_c4) + 1.264747*(del_Ue*fgd)**2.369401)**H)**(-1.0)
    # division by 4pi for double sided wavenumber spectra
    # phipp = (C1+C2)/(C3+C4)/(4*np.pi)
    # division by 4pi for double sided wavenumber spectra
    # division by 4pi for double sided wavenumber spectra
    phipp = (C1+C2)/(C3+C4)/(4*np.pi)
    # phipp = C5/(4*np.pi)

    
    if (isinf(phipp)).any():
      print('Inf found: ',where(isinf(phipp))[0][0],where(isinf(phipp))[1][0])
      raise RuntimeError("Error in GEP-2-mod model")
      
    if (isnan(phipp)).any():
      print('NaN found: ',where(isnan(phipp))[0][0],where(isnan(phipp))[1][0])
      raise RuntimeError("Error in GEP-2-mod model")
    return phipp



def _phipp_Pargal(istrip,flow_params,omega):
    from numpy import isnan,where,isinf
    import numpy as np
    U_X,rho,q=_flow_param(istrip,flow_params)

    delta99=flow_params['delta99'][istrip]
    delta_star=flow_params['delta_star'][istrip]
    tau_wall=flow_params['tau_wall'][istrip]
    theta=flow_params['theta'][istrip]   

    if tau_wall<1.0e-3:
        separation = True
    else:
        separation = False

    # hard coded 
    kinematic_viscosity=1.56e-5
    Re_theta = U_X * theta / kinematic_viscosity

    pressure_gradient=max(flow_params['dpds'][istrip],0.)
    max_uv = flow_params['uv_max'][istrip]

    beta_c = (theta/tau_wall)*pressure_gradient #Clauser' parameter
    PI = 0.8*(beta_c+0.5)**0.75 #Law of the wake parameter (Durbi & Reif empirical formula) 

    # For now we don't do ym detection
    ym = 0

    C1 = 0.7
    C2 = 3.0

    # Setting constrained for low Re cases as well as extreme
    if (ym < 15.0) or (Re_theta < 300):
        ym = 15.0
        C3 = 0.8
    else:
        mul=(ym**0.7575) * (Pi**1.864)
        C3 = (3.34e-4*mul)+0.8 

    # Setting constrained for separated flow
    if separation:
        ym = 2.0
        C3 = 0.75

    C4 = ym**(-0.365)

    strouhal_number=_compute_strouhal(U_X,delta99,omega)  

    ampl = 0.5 * max_uv**2 * delta99  / U_X

    phipp = ampl * (C2 * strouhal_number**2) / ((strouhal_number**C3 + C1)**3.7 + (C4 * strouhal_number)**7 )
    
    return phipp