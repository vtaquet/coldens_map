#################################################################################
# coldens_map.py
# Routine that creates total column density and rotational temperature maps from 
# a series of moment 0 maps of the same molecule using a standard rotational 
# diagram analysis (LTE analysis, optically thin emission).
#################################################################################

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import leastsq
from numpy import array, arange, sin
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from reproject import reproject_interp
import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests
import pandas as pd
import os

# Population diagram analysis (here optically thin emission)
def pd_analysis(row, df_spec, rms, choice_plot):

  Ntot = 0. ; Trot = 0. ; dNtot = 0. ; dTrot = 0.
  # assess whether lines are detected
  if choice_y == 'kelvin':
    intens = np.array([row['Tmb_%.3f' % freq] for freq in df_spec["nu"]])
  elif choice_y == 'jansky':
    intens = np.array([row['Flux_%.3f' % freq] for freq in df_spec["nu"]])
  detect = [intens[i] > threshold*rms[i] for i in range(len(rms))]
  #
  # continue calculation if intensity of all transitions are above 3rms
  if sum(detect) >= len(rms)*0.8:
    #print(row['lon'],row['lat'])
    # extract useful variables for RD
    gup = df_spec.ix[detect,'gup'].values
    Eup = df_spec.ix[detect,'Eup']/Ktocm
    Nup = np.array([row['Nup_%.3f' % freq] for freq in df_spec.ix[detect,"nu"]])/gup
    dNup = np.array([row['dNup_%.3f' % freq] for freq in df_spec.ix[detect,"nu"]])/Nup
    Nup = np.log(Nup) # assuming beamdil = 1 for now
    #
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    #
    # fit the rotational diagram
    pinit = [1.0, -1.0]
    out = leastsq(errfunc, pinit,args=(Eup, Nup, dNup), full_output=1)
    pfinal = out[0] ; covar = out[1]
    # derive Ntot and Trot from best-fit values
    Ntot = np.exp(pfinal[0]) ; Trot = -1/pfinal[1]
    pf_Trot = float(pf_interp(Trot))
    Ntot = Ntot*pf_Trot
    # derive uncertainties
    dNtot = np.sqrt(covar[0][0]) * Ntot ; dTrot = np.sqrt(covar[1][1])/np.abs(pfinal[1]) * Trot
    #
    # plot rotational diagram for Ntot peak
    if choice_plot == 'yes':
        legend = '$N_{tot}$ = %.1e $\pm$ %.1e cm$^{-2}$' % (Ntot,dNtot)
        legend2 = '$T_{rot}$ = %.1f $\pm$ %.1f K' % (Trot,dTrot)
        plt.clf()
        plt.figure(0)
        Euplim = np.array([0,np.max(Eup)*1.2])
        plt.plot(Euplim, fitfunc(pfinal,Euplim))
        plt.errorbar(Eup, Nup, yerr=dNup, fmt='k.')  # Data
        plt.xlabel('E$_{up}$ [K]')
        plt.ylabel('ln($N_{up}/g_{up}$)')
        plt.xlim(0,np.max(Eup)*1.2)
        plt.ylim(np.min(Nup)*0.95,np.max(Nup)*1.05)
        plt.text(0.5*np.max(Eup),1.04*np.max(Nup),legend)
        plt.text(0.5*np.max(Eup),1.03*np.max(Nup),legend2)
        namefile = 'popdiag_lon=%.2f_lat=%.2f' % (row['lon'],row['lat'])
        plt.savefig('figures/'+prefix+namefile+'.eps',bbox_inches='tight')
        plt.close(0)
        #print(Ntot,dNtot)
        #print(Trot,dTrot)
  #      
  row['Ntot'] = Ntot ; row['dNtot'] = dNtot ; row['Trot'] = Trot ; row['dTrot'] = dTrot

  return row

##------------------------------------
##------------------------------------
## constants
##------------------------------------
##------------------------------------

c = 2.99e10 ; kb = 1.3803e-16 ; hb = 6.62e-27 
Ktocm = 0.695 ; JtoeV = 1.602e-19 ; cmtoeV = 8065.54 
iglob = 0.

##---------------------------------
##---------------------------------
## read input files
##---------------------------------
##---------------------------------

print("Reading input files...")

# check if directories exist
if os.path.isdir('predictions') == False:
  os.system("mkdir predictions")
if os.path.isdir('data') == False:
  os.system("mkdir data")
if os.path.isdir('figures') == False:
  os.system("mkdir figures")
if os.path.isdir('data/cdms') == False:
  os.system("mkdir data/cdms")
if os.path.isdir('data/jpl') == False:
  os.system("mkdir data/jpl")

# read input file and store input params into dictionary
inpfile = "input.in"
inpascii = open(inpfile,'r')
inpparam = [] ; inpvalue = []
for line in inpascii:
  s = line.split()
  inpparam.append(s[0])
  inpvalue.append(s[2])
inp_dict = dict(zip(inpparam,inpvalue))

# assign input variables
namespec = inp_dict["species"]            # Species to consider
choice_data = inp_dict["choice_data"]     # Where to look for spectro data (local or online)
fileobs = inp_dict["fileobs"]             # Observation file
source_lon = inp_dict["source_lon"]       # Longitude of reference source coordinates
source_lat = inp_dict["source_lat"]       # Latitude of reference source coordinates
choice_y = inp_dict["choice_y"]           # Unit of intensity value: Tmb (K), Fpeak (Jy)
prefix   = inp_dict["prefix"]             # Prefix for output files
datab = inp_dict["database"]              # Spectroscopic database: jpl or cdms
threshold = float(inp_dict["threshold"])  # Threshold in steps of rms over which transitions are included in the RD
#choice_plot = int(inp_dict["choice_plot"])# Output for the plot: 1) pdf, 2) python window

sourcecoord = SkyCoord(source_lon,source_lat,frame='icrs', unit="deg")

# read obs file
inpobs = open(fileobs,'r')
freqobs = [] ; fitsobs = [] ; rmsobs = [] ; beam1obs = [] ; beam2obs = []
for line in inpobs:
  s = line.split()
  freqobs.append(float(s[0])*1e3)
  fitsobs.append(s[1])
  rmsobs.append(float(s[2]))
  beam1obs.append(float(s[3]))
  beam2obs.append(float(s[4]))
Nlines = len(freqobs)

# read fits files
fitshead = [] ; fitsdata = [] ; fitswcs = [] ; source_x = [] ; source_y = [] ; data_rescaled = [] ; 
lon_1d = [] ; lat_1d = [] ; lon_2d = [] ; lat_2d = [] ; x_1d = [] ; y_1d = [] ; x_2d = [] ; y_2d = []
pix_ra = [] ; pix_dec = [] ; unit = [] ; Nra = [] ; Ndec = [] 
for ifits, ffile in enumerate(fitsobs):
  fitsopen = fits.open(ffile)
  #
  # remove dummy dimensions if they exist
  if fitsopen[0].header['CTYPE3'].replace(" ", "") == 'UNKNOWN' or fitsopen[0].header['CTYPE3'].replace(" ", "") == '':
    del fitsopen[0].header['CTYPE3'] ; del fitsopen[0].header['CRVAL3'] ; del fitsopen[0].header['NAXIS3']
    del fitsopen[0].header['CDELT3'] ; del fitsopen[0].header['CRPIX3'] ; del fitsopen[0].header['CROTA3'] 
    fitsopen[0].header['NAXIS'] += -1
  if fitsopen[0].header['CTYPE4'].replace(" ", "") == 'UNKNOWN' or fitsopen[0].header['CTYPE4'].replace(" ", "") == '':
    del fitsopen[0].header['CTYPE4'] ; del fitsopen[0].header['CRVAL4'] ; del fitsopen[0].header['NAXIS4']
    del fitsopen[0].header['CDELT4'] ; del fitsopen[0].header['CRPIX4'] ; del fitsopen[0].header['CROTA4'] 
    fitsopen[0].header['NAXIS'] += -1
  #
  # extract datacube according to array shape
  if len(fitsopen[0].data.shape) == 2:
    data2 = fitsopen[0].data
  elif len(fitsopen[0].data.shape) == 3:
    data2 = fitsopen[0].data[0,:,:]
  elif len(fitsopen[0].data.shape) == 4:
    data2 = fitsopen[0].data[0,0,:,:]
  data2[np.isnan(data2)] = 0.
  #
  # create header and datacubes
  fitshead.append(fitsopen[0].header)
  fitsdata.append(data2)
  fitswcs.append(WCS(fitshead[ifits]))
  #
  # create useful variables from header
  Nra.append(fitshead[ifits]['NAXIS1'])
  Ndec.append(fitshead[ifits]['NAXIS2'])
  unit.append(fitshead[ifits]['BUNIT'])
  #
  # create coordinate tables with "world" coordinates
  x1d = np.arange(0,Nra[ifits], 1) ; y1d = np.arange(0,Ndec[ifits], 1)
  x2d, y2d = np.meshgrid(x1d, y1d)
  lon2, lat2 = fitswcs[ifits].all_pix2world(x2d,y2d, 0)
  lon1 = lon2[0,:]
  lat1 = lat2[:,0]
  lon_1d.append(lon1) ; lat_1d.append(lat1)
  lon_2d.append(lon2) ; lat_2d.append(lat2)
  #
  # source coordinate to "pixel" coordinates
  x0, y0 = fitswcs[ifits].wcs_world2pix(sourcecoord.ra,sourcecoord.dec, 1)
  source_x.append(x0) ; source_y.append(y0)
  #
  # change dimensions of datacube to match dimensions of 1st fits through interpolation
  data_newsize, data_footprint = reproject_interp((data2, fitswcs[ifits]), fitshead[0])
  data_rescaled.append(data_newsize)
  if ifits >= 1:
    list_val = [float(data_newsize[idec,ira]) for idec in range(Ndec[0]) for ira in range(Nra[0]) ]
    if choice_y == 'kelvin':
      df_obsdata['Tmb_%.3f' % (float(freqobs[ifits]))] = [float(dat) for dat in np.nditer(data_newsize)]  #list_val 
    elif choice_y == 'jansky':
      df_obsdata['Flux_%.3f' % (float(freqobs[ifits]))] = [float(dat) for dat in np.nditer(data_newsize)]  #list_val 
  elif ifits == 0:
    df_obsdata = pd.DataFrame({
               'lon' : [float(x) for x in np.nditer(lon_2d[ifits])],
               'lat' : [float(y) for y in np.nditer(lat_2d[ifits])]
               })
    if choice_y == 'kelvin':
      df_obsdata['Tmb_%.3f' % (float(freqobs[ifits]))] = [float(dat) for dat in np.nditer(data2)] 
    elif choice_y == 'jansky':
      df_obsdata['Flux_%.3f' % (float(freqobs[ifits]))] = [float(dat) for dat in np.nditer(data2)] 
#
#
##------------------------------------
##------------------------------------
## read partition functions data
##------------------------------------
##------------------------------------
#
# read catdir from JPL and CDMS websites or locally
catdir_jpl = "https://spec.jpl.nasa.gov/ftp/pub/catalog/catdir.cat"
catdir_cdms = "http://www.astro.uni-koeln.de/site/vorhersagen/catalog/partition_function.html"
#
if choice_data == 'online': # read the catdir online
  resp_jpl = requests.get(catdir_jpl)
  catdir_file_jpl = resp_jpl.text.split("\n")
  del(catdir_file_jpl[-1])
  resp_cdms = requests.get(catdir_cdms)
  catdir_file_cdms = resp_cdms.text.split("\n")
  catdir_file_cdms = [line.replace("---", "0.0") for line in catdir_file_cdms]
  del(catdir_file_cdms[0:14])
  del(catdir_file_cdms[-5:-1])
  del(catdir_file_cdms[-1])
  # write the catdir files locally
  with open("data/catdir_jpl.cat", 'w') as catjpl:
    for line in catdir_file_jpl:
      catjpl.write(line+'\n')
  with open("data/catdir_cdms.cat", 'w') as catcdms:
    for line in catdir_file_cdms:
      catcdms.write(line+'\n')
elif choice_data == 'local': # read the catdir locally
  catdir_file_jpl = [] ; catdir_file_cdms = []
  with open("data/catdir_jpl.cat",'r') as catjpl:
    for line in catjpl.readlines():
      catdir_file_jpl.append(line[0:-1])
  with open("data/catdir_cdms.cat",'r') as catcdms:
    for line in catcdms.readlines():
      catdir_file_cdms.append(line[0:-1])

# create dictionary and panda frames
catdir_dic_jpl = {'tag': [int(line[0:7]) for line in catdir_file_jpl], 
              'species': [line[7:20].replace(" ","") for line in catdir_file_jpl], 
              'Nl':      [int(line[20:27]) for line in catdir_file_jpl],
              'T_300K':  [pow(10,float(line[27:33])) for line in catdir_file_jpl],
              'T_225K':  [pow(10,float(line[34:40])) for line in catdir_file_jpl],
              'T_150K':  [pow(10,float(line[41:47])) for line in catdir_file_jpl],
              'T_75K':   [pow(10,float(line[48:54])) for line in catdir_file_jpl],
              'T_37.5K': [pow(10,float(line[55:61])) for line in catdir_file_jpl],
              'T_18.75': [pow(10,float(line[62:68])) for line in catdir_file_jpl],
              'T_9.375K':[pow(10,float(line[69:75])) for line in catdir_file_jpl]}
catdir_dic_cdms = {'tag':[int(line[0:7]) for line in catdir_file_cdms], 
              'species': [line[7:33].replace(" ","") for line in catdir_file_cdms], 
              'Nl':      [int(line[33:39]) for line in catdir_file_cdms],
              'T_1000K': [pow(10,float(line[46:52])) for line in catdir_file_cdms],
              'T_500K':  [pow(10,float(line[59:65])) for line in catdir_file_cdms],
              'T_300K':  [pow(10,float(line[71:78])) for line in catdir_file_cdms],
              'T_225K':  [pow(10,float(line[84:91])) for line in catdir_file_cdms],
              'T_150K':  [pow(10,float(line[97:104])) for line in catdir_file_cdms],
              'T_75K':   [pow(10,float(line[110:117])) for line in catdir_file_cdms],
              'T_37.5K': [pow(10,float(line[123:130])) for line in catdir_file_cdms],
              'T_18.75': [pow(10,float(line[136:143])) for line in catdir_file_cdms],
              'T_9.375K':[pow(10,float(line[149:156])) for line in catdir_file_cdms],
              'T_5K':    [pow(10,float(line[162:169])) for line in catdir_file_cdms],
              'T_2.725K':[pow(10,float(line[175:183])) for line in catdir_file_cdms]}

catdir_pd_jpl = pd.DataFrame(catdir_dic_jpl)
catdir_pd_jpl.index = [catdir_pd_jpl["species"]]
catdir_pd_cdms = pd.DataFrame(catdir_dic_cdms)
catdir_pd_cdms.index = [catdir_pd_cdms["species"]]

        
##----------------------------------------
##----------------------------------------
## read spectro data from JPL/CDMS
##----------------------------------------
##----------------------------------------

bd = 1.

print("Reading "+namespec+"'s spectroscopic data...")

# read spectro file from JPL or CDMS website
if datab == 'jpl':
  try:
    pf = catdir_pd_jpl.loc[[namespec],['T_300K','T_225K','T_150K','T_75K','T_37.5K','T_18.75','T_9.375K']].values.tolist()[0]
  except:
    print("%15s is not in the JPL database: STOP" % (namespec))
    exit()
  Tpf = [300., 225., 150., 75., 37.5, 18.75, 9.375] ; Npf = len(Tpf)
  specurl = "https://spec.jpl.nasa.gov/ftp/pub/catalog/c%06i.cat" % (catdir_pd_jpl.loc[namespec,"tag"])
  specpath = 'data/'+datab+'/c%06i.cat' % (catdir_pd_jpl.loc[namespec,"tag"])
elif datab == 'cdms':
  try:
    pf = catdir_pd_cdms.loc[[namespec],['T_1000K','T_500K','T_300K','T_225K','T_150K','T_75K','T_37.5K','T_18.75','T_9.375K','T_5K','T_2.725K']].values.tolist()[0]
  except:
    print("%15s is not in the CDMS database: STOP" % (namespec))
    exit()
  Tpf = [1000., 500., 300., 225., 150., 75., 37.5, 18.75, 9.375, 5., 2.725] ; Npf = len(Tpf)
  #print(str(catdir_pd_cdms.loc[namespec,"tag"]).zfill(6))
  specurl = "http://www.astro.uni-koeln.de/site/vorhersagen/catalog/c%06i.cat" % (catdir_pd_cdms.loc[namespec,"tag"])
  specpath = 'data/'+datab+'/c%06i.cat' % (catdir_pd_cdms.loc[namespec,"tag"])
  # assign pf values to closest ones when there is no data    
  for it in range(Npf):
      if pf[it] == 1.0:
          if it > 1:
              if pf[it-1] > 1.0:
                  pf[it] = pf[it-1]
              elif pf[it-1] == 1.0:
                  pf[it] = pf[it-2]
          else:
              if pf[it+1] > 1.0:
                  pf[it] = pf[it+1]
              elif pf[it+1] == 1.0:
                  pf[it] = pf[it+2]

# compute partition function
pf_interp = interp1d(Tpf,pf,bounds_error=False,fill_value='extrapolate')
Zint = float(pf_interp(300.)) # pf at 300 K

# read spectro datafile online or locally
if choice_data == 'online':
  response = requests.get(specurl)
  specfile = response.text.split("\n")
  with open(specpath, 'w') as catfile:
    for line in specfile:
      catfile.write(line+'\n')
elif choice_data == 'local':
  specfile = []
  with open(specpath,'r') as catfile:
    for line in catfile.readlines():
      specfile.append(line[0:-1])
del(specfile[-1])

# create dictionary and panda frame from spectro file
try:
  spec_pd2 = pd.DataFrame(
             {'species' : [namespec for line in specfile], 
              'nu':       [float(line[0:13]) for line in specfile], 
              'dnu':      [float(line[13:20]) for line in specfile],
              'intens':   [10**float(line[21:29]) for line in specfile],
              'Elow':     [float(line[31:41]) for line in specfile],
              'gup':      [float(line[41:44]) for line in specfile],
              'line':     [(line[44:-1]) for line in specfile],
              'bd':       [bd for line in specfile],
              'Zint':     [Zint for line in specfile]})
except: # gup is not a int in some CDMS files
  spec_pd2 = pd.DataFrame(
             {'species' : [namespec for line in specfile], 
              'nu':       [float(line[0:13]) for line in specfile], 
              'dnu':      [float(line[13:20]) for line in specfile],
              'intens':   [10**float(line[21:29]) for line in specfile],
              'Elow':     [float(line[31:41]) for line in specfile],
              'gup':      [float(line[42:44]) for line in specfile],
              'line':     [(line[44:-1]) for line in specfile],
              'bd':       [bd for line in specfile],
              'Zint':     [Zint for line in specfile]})
#
spec_pd = spec_pd2[np.isclose(spec_pd2.nu.values[:,None],freqobs,rtol=1e-10).any(axis=1)].reset_index()
#
# compute intensities from spectro properties
spec_pd["Eup"] = spec_pd["Elow"] + hb*1e-7*spec_pd["nu"]*1e6*cmtoeV/JtoeV
spec_pd["Aij"] = spec_pd["intens"]*((spec_pd["nu"])**2)*spec_pd["Zint"]/spec_pd["gup"]*\
                 (np.exp(-spec_pd["Elow"]/Ktocm/3e2)-np.exp(-spec_pd["Eup"]/Ktocm/3e2))**(-1)*2.7964e-16
#
# compute Nup in all pixels for each transition
Nup = []
for ifits, fdata in enumerate(fitsdata):
  # compute flux or main-beam temperature according to choice_y
  if choice_y == 'kelvin':
    df_obsdata['Flux_%.3f' % (float(freqobs[ifits]))] = (1.222e6)**(-1)*beam1obs[ifits]*beam2obs[ifits]*(spec_pd.ix[ifits,"nu"]/1e3)**2*df_obsdata['Tmb_%.3f' % (float(freqobs[ifits]))]
    df_obsdata['dFlux_%.3f' % (float(freqobs[ifits]))] = (1.222e6)**(-1)*beam1obs[ifits]*beam2obs[ifits]*(spec_pd.ix[ifits,"nu"]/1e3)**2*rmsobs[ifits]
  elif choice_y == 'jansky':
    #print(df_obsdata.ix[0:5,'Flux_%.3f' % (float(freqobs[ifits]))] / ((1.222e6)**(-1)*beam1obs[ifits]*beam2obs[ifits]*(spec_pd.ix[ifits,"nu"]/1e3)**2))
    df_obsdata['Tmb_%.3f' % (float(freqobs[ifits]))] = df_obsdata['Flux_%.3f' % (float(freqobs[ifits]))] / ((1.222e6)**(-1)*beam1obs[ifits]*beam2obs[ifits]*(spec_pd.ix[ifits,"nu"]/1e3)**2)
    df_obsdata['dTmb_%.3f' % (float(freqobs[ifits]))] = rmsobs[ifits] / ((1.222e6)**(-1)*beam1obs[ifits]*beam2obs[ifits]*(spec_pd.ix[ifits,"nu"]/1e3)**2)
  # compute Nup from Tmb
  df_obsdata['Nup_%.3f' % (float(freqobs[ifits]))] = 8.*np.pi*kb*(spec_pd.ix[ifits,"nu"]*1e6)**2*df_obsdata['Tmb_%.3f' % (float(freqobs[ifits]))]*1e5/(hb*c**3*spec_pd.ix[ifits,"Aij"])
  df_obsdata['dNup_%.3f' % (float(freqobs[ifits]))] = 8.*np.pi*kb*(spec_pd.ix[ifits,"nu"]*1e6)**2*rmsobs[ifits]*1e5/(hb*c**3*spec_pd.ix[ifits,"Aij"])


################################################################################
################################################################################
## deduce rotational temperatures, and column densities with a RD analysis
################################################################################
################################################################################

print("Performing rotational diagram analysis...")

#
# apply RD analysis to each pixel in dataframe
df_obsdata = df_obsdata.apply(pd_analysis,axis=1,df_spec=spec_pd,rms=rmsobs,choice_plot='no')
#
# convert dataframe to np array
np_Ntot = np.array([[df_obsdata.ix[iy*Nra[0]+ix,"Ntot"] for ix in range(Nra[0])] for iy in range(Ndec[0])])
np_Trot = np.array([[df_obsdata.ix[iy*Nra[0]+ix,"Trot"] for ix in range(Nra[0])] for iy in range(Ndec[0])])
np_dNtot = np.array([[df_obsdata.ix[iy*Nra[0]+ix,"dNtot"] for ix in range(Nra[0])] for iy in range(Ndec[0])])
np_dTrot = np.array([[df_obsdata.ix[iy*Nra[0]+ix,"dTrot"] for ix in range(Nra[0])] for iy in range(Ndec[0])])
#
# write dataframe to csv
df_obsdata.to_csv('figures/'+prefix+'dataframe.csv')

####################
####################
## Make the plots ##
####################
####################

print("Generating the plots...")

#
# Ticks
#dxa = dpix*Npix
Nticks = 11 #int(dxa/5.) + 1
Nticks2 = int(Nticks/2.)*1.
#
hfont = {'family' : 'serif',
         'serif' : 'Verdana',
         'weight' : 'medium',
         'size'   : 12}
haxes = {'linewidth' : 1.5}
hticks = {'major.size' : 6,
          'major.width' : 1.5,
          'minor.size' : 3,
          'minor.width' : 1}
plt.rc('font', **hfont)
plt.rc('axes',**haxes)
plt.rc('xtick',**hticks)
plt.rc('ytick',**hticks)
plt.rcParams['contour.negative_linestyle'] = 'solid'
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
#
# plot maps
#
x_max = +200 ; x_min = -300 ; y_max = 500. ; y_min = -200
x_tick_lab = np.array([200,0,-200])#,-400])
y_tick_lab = np.array([-200,0,200,400])#,600])
#
# plot emission maps
for ifits, ffile in enumerate(fitsobs):
  #
  # plot original maps
  title_bar = 'Intensity ['+unit[ifits]+']'
  fig = plt.figure(0)
  ax = fig.add_subplot(111, projection=fitswcs[ifits])
  ax.coords[0].set_major_formatter('hh:mm:ss')
  ax.coords[1].set_major_formatter('dd:mm:ss')
  ax.set_xlabel('R.A. [J2000]')
  ax.set_ylabel('Dec. [J2000]')
  cax = ax.imshow(fitsdata[ifits],origin='lower',cmap='inferno',interpolation='nearest',vmin=-0.1*np.nanmax(fitsdata[ifits]),vmax=np.nanmax(fitsdata[ifits])) 
  ax.contour(fitsdata[ifits],levels=[i*3*rmsobs[ifits] for i in range(100)],colors='black')
  fig.colorbar(cax, label=title_bar)
  namefile = "%.0f" % freqobs[ifits] 
  plt.savefig('figures/'+prefix+namefile+'.eps',bbox_inches='tight')
  plt.close(0)
  #
  # plot re-scaled maps
  title_bar = 'Intensity ['+unit[ifits]+']'
  fig = plt.figure(0)
  ax = fig.add_subplot(111, projection=fitswcs[0])
  ax.coords[0].set_major_formatter('hh:mm:ss')
  ax.coords[1].set_major_formatter('dd:mm:ss')
  ax.set_xlabel('R.A. [J2000]')
  ax.set_ylabel('Dec. [J2000]')
  cax = ax.imshow(data_rescaled[ifits],origin='lower',cmap='inferno',interpolation='nearest',vmin=-0.1*np.nanmax(data_rescaled[ifits]),vmax=np.nanmax(data_rescaled[ifits])) 
  ax.contour(data_rescaled[ifits],levels=[i*3*rmsobs[ifits] for i in range(100)],colors='black')
  fig.colorbar(cax, label=title_bar)
  namefile = "%.0f" % freqobs[ifits] 
  plt.savefig('figures/'+prefix+namefile+'_rescaled.eps',bbox_inches='tight')
  plt.close(0)
#
# plot Ntot
title_bar = 'Ntot'
fig = plt.figure(0)
ax = fig.add_subplot(111, projection=fitswcs[0])
ax.coords[0].set_major_formatter('hh:mm:ss')
ax.coords[1].set_major_formatter('dd:mm:ss')
ax.set_xlabel('R.A. [J2000]')
ax.set_ylabel('Dec. [J2000]')
cax = ax.imshow(np_Ntot,origin='lower',cmap='inferno',interpolation='nearest',vmin=-0.1*np.nanmax(np_Ntot),vmax=np.nanmax(np_Ntot)) 
ax.contour(np_Ntot,levels=[i*0.1*np.nanmax(np_Ntot) for i in range(10)],colors='black')
fig.colorbar(cax, label=title_bar)
namefile = "%.0f" % freqobs[ifits] 
plt.savefig('figures/'+prefix+'Ntot.eps',bbox_inches='tight')
plt.close(0)
#
# plot dNtot
title_bar = 'dNtot'
fig = plt.figure(0)
ax = fig.add_subplot(111, projection=fitswcs[0])
ax.coords[0].set_major_formatter('hh:mm:ss')
ax.coords[1].set_major_formatter('dd:mm:ss')
ax.set_xlabel('R.A. [J2000]')
ax.set_ylabel('Dec. [J2000]')
cax = ax.imshow(np_dNtot,origin='lower',cmap='inferno',interpolation='nearest',vmin=-0.1*np.nanmax(data_rescaled[ifits]),vmax=np.nanmax(np_dNtot)) 
fig.colorbar(cax, label=title_bar)
namefile = "%.0f" % freqobs[ifits] 
plt.savefig('figures/'+prefix+'dNtot.eps',bbox_inches='tight')
plt.close(0)
#
# plot Trot
title_bar = 'Trot'
fig = plt.figure(0)
ax = fig.add_subplot(111, projection=fitswcs[0])
ax.coords[0].set_major_formatter('hh:mm:ss')
ax.coords[1].set_major_formatter('dd:mm:ss')
ax.set_xlabel('R.A. [J2000]')
ax.set_ylabel('Dec. [J2000]')
cax = ax.imshow(np_Trot,origin='lower',cmap='inferno',interpolation='nearest',vmin=5,vmax=np.nanmax(np_Trot)) 
fig.colorbar(cax, label=title_bar)
namefile = "%.0f" % freqobs[ifits] 
plt.savefig('figures/'+prefix+'Trot.eps',bbox_inches='tight')
plt.close(0)
#
# plot dTrot
title_bar = 'dTrot'
fig = plt.figure(0)
ax = fig.add_subplot(111, projection=fitswcs[0])
ax.coords[0].set_major_formatter('hh:mm:ss')
ax.coords[1].set_major_formatter('dd:mm:ss')
ax.set_xlabel('R.A. [J2000]')
ax.set_ylabel('Dec. [J2000]')
cax = ax.imshow(np_Trot,origin='lower',cmap='inferno',interpolation='nearest',vmin=-0.1*np.nanmax(np_dTrot),vmax=np.nanmax(np_dTrot)) 
fig.colorbar(cax, label=title_bar)
namefile = "%.0f" % freqobs[ifits] 
plt.savefig('figures/'+prefix+'dTrot.eps',bbox_inches='tight')
plt.close(0)
#
# plot rotational diagram toward Ntot peak
row_Ntotmax = pd_analysis(df_obsdata.iloc[df_obsdata["Ntot"].idxmax()],df_spec=spec_pd,rms=rmsobs,choice_plot='yes')

exit()


