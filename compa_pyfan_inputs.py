import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 12

files = dict()
# files['powerflow'] = 'input_H380_TE_PF_smooth.txt'
files['2500m3h'] = 'RANS_H380EC01_5M_CFD_import_2500m3h_rotor_TE.dat'
files['3500m3h'] = 'RANS_H380EC01_5M_CFD_import_3500m3h_rotor_TE.dat'

def read_input(pyfan_input):
    df = pd.read_csv(pyfan_input,sep=' ',skiprows=1)
    with open(pyfan_input,'r') as fio:
        print(pyfan_input)
        header = fio.readline().strip().split(' ')[1:]
        fio.close()
    df.columns = header
    nu = 1.5e-5
    df['Delta'] = df['bl_thickness']/df['displacement_thickness']
    df['H'] = df['displacement_thickness']/df['momentum_thickness']
    df['ReTheta'] = df['momentum_thickness'] * df['exterior_stream_velocity']/nu
    df['Q'] = 0.5* df['exterior_density'] * df['exterior_stream_velocity']**2
    df['u_tau'] = (df['tau_wall']/df['exterior_density'])**0.5
    df['RT'] = df['bl_thickness']/df['exterior_stream_velocity']/(nu/df['u_tau']**2)
    df['betaC'] = df['displacement_thickness']/df['tau_wall']*df['dpds']
    df['Pi'] = 0.8*(df['betaC']+0.5)**0.75
    df['Cf'] = df['tau_wall']/df['Q']
    df['Cuv'] = df['uv_max']/df['Q']
    df['Rpct'] = 100*(df['radius']-0.078)/(0.183-0.078)
    return df

data = dict()
for key in files.keys():
    data[key] = read_input(files[key])

fig,ax = plt.subplots(1,7,figsize=(15,5),sharey=True,sharex=False)
iplot = 0
for var,scale,label in zip(['Delta','H','betaC','ReTheta','RT','Cf','Cuv'],
                           [1.,1.,1.,1.,1.,1000.,1000.],
                           [r'$\Delta$',r'$H$',r'$\beta_C$',r'$Re_\theta$',r'$Re_T$',r'$C_f\times10^3$',r'$C_{uv}\times10^3$']):
    for key,name,col in zip(data.keys(),[r'2500 m$^3$/h',r'3500 m$^3$/h'],['blue','red']):
        ax[iplot].plot(scale*data[key][var],data[key]['Rpct'],label=name,color=col)
    ax[iplot].set_xlabel(label)
    ax[iplot].grid()
    iplot += 1
    # plt.legend()
ax[0].set_ylabel('Radius [%]')
ax[6].legend(loc=2,bbox_to_anchor=(0.5,0.5))
ax[1].set_xlim(None,4)
ax[2].set_xlim(None,20)
ax[3].set_xlim(None,1000)
ax[5].set_xlim(None,20)
ax[6].set_xlim(None,40)
plt.savefig('figure_BLparameters.pdf')
plt.show()