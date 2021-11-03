## SIFDM Data Interpretations:
##
## Created by Connor Painter on 9/27/21 to do elementary analysis on the 
## outputs of SIFDM MATLAB code by Philip Mocz with Python.
## Last updated: 10/28/21



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import h5py
import os
import time
import itertools
from scipy.optimize import curve_fit
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'



## GETTING STARTED: 
##  -   Prepare to import your data (from Philip's .h5 output) by listing paths (strings) to output folders.
##      -   List must be named 'folders'.
##      -   Feel free to use my formatting below.
##  -   Set saving preferences with 'createImgSubfolders' and 'saveToParent' below.
##      -   'saveToParent' is the parent directory to which you want images saved.
##      -   If 'createImgSubfolders' is True, compiling code will create subfolders to organize saved images.



## Folder setup


## f15 = Inf; L = 20; T = 4; Nout = 400; N (res) = 100
d1 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/fInfL20T4n400r100"

## f15 = 4; L = 20; T = 4; Nout = 400; N (res) = 100
d2 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f4L20T4n400r100"

## f15 = 2; L = 20; T = 4; Nout = 400; N (res) = 100
d3 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f2L20T4n400r100"

## f15 = 1; L = 20; T = 4; Nout = 400; N (res) = 100
d4 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f1L20T4n400r100"




folders = [d1, d2, d3, d4]



## Preferences for saved images



createImgSubfolders = True
saveToParent = os.getcwd()



## Useful arrays and dictionaries



Q = {'':'', 
     'snapdir':'Full Path of Snap-Enclosing Directory', 
     'num':'Snap Number', 'filename':'Snap File Name', 
     'path':'Full Path of Snap', 
     'dir':'Snap-Enclosing Directory', 
     'psi':'Wavefunction', 
     't':'Current Time', 
     'm22':'Mass [10^-22 eV]', 
     'm':'Mass', 
     'f15':'Strong-CP Energy Decay Constant [10^15 GeV]', 
     'f':'Strong-CP Energy Decay Constant', 
     'a_s':'s-scattering Length', 
     'critical_M_sol':'Critical Soliton Mass', 
     'critical_rho0':'Critical Central Soliton Density', 
     'critical_r_c':'Critical Soliton Core Radius',
     'critical_beta':'Critical Beta',
     'Lbox':'Simulation Box Length', 
     'N':'Resolution', 
     'dx':'Grid Spacing',
     'phase':'Wavefunction Phase', 
     'rho':'Density', 
     'rhobar':'Mean Density', 
     'rho0':'Central Soliton Density', 
     'i0':'Maximum Density Index', 
     'beta':'Soliton Stability Constant (Beta)',
     'r_c':'Soliton Core Radius', 
     'M_sol':'Soliton Mass', 
     'critical_f':'Critical Strong-CP Energy Decay Constant for Collapse',
     'tailindex':'Index of Power Law Fit to Density Profile Tail', 
     'V':'Potential', 
     'v':'Madelung Velocity', 
     'v2':'Madelung Velocity Magnitude Squared', 
     'M':'Total Mass', 
     'W':'Potential Energy', 
     'Kv':'Classical Kinetic Energy', 
     'Krho':'Quantum Gradient Energy', 
     'KQ':'Total Kinetic Energy',
     'E':'Total Energy',
     'L':'Angular Momentum', 
     'L2':'Angular Momentum Magnitude Squared'}
Q0 = ['t', 'm22', 'm', 'f15', 'f', 'a_s', 'critical_M_sol', 'critical_rho0', 'critical_r_c', 'critical_beta', 'Lbox', 'N', 'dx', 'rhobar', 'rho0', 'beta', 'r_c', 'M_sol', 'critical_f', 'tailindex', 'M', 'W', 'Kv', 'Krho', 'KQ', 'L2']
Q1 = ['L', 'i0']
Q2 = []
Q3 = ['psi', 'phase', 'rho', 'V', 'v2']
Q4 = ['v']
U = {'length':'[kpc]', 'mass':r"[$M_{\odot}$]", 'energy':r"[$M_{\odot}\mathrm{(km/s)}^2$]", 'time':"[kpc/(km/s)]", 'density':r"[$M_{\odot}/\mathrm{kpc}^3$]", 'velocity':"[km/s]",
     't':'time', 'm':'mass', 'f':'energy', 'Lbox':'length', 'dx':'length', 'rho':'density', 'rhobar':'density', 'rho0':'density', 'r_c':'length', 'M_sol':'mass', 'E':'energy', 'W':'energy', 'Kv':'energy', 'Krho':'energy', 'KQ':'energy'}
C = {'phase':'bwr', 'rho':'inferno', 'V':'cividis', 'v':'hot', 'v2':'hot'}
LTX = {'psi':r"$\Psi$",
       't':r"$t$",
       'm22':r"$\frac{m}{(10^{-22} \mathrm{eV})}$",
       'm':r"$m$",
       'f15':r"$\frac{f}{(10^{15} \mathrm{eV})}$",
       'f':r"$f$",
       'a_s':r"$a_s$",
       'critical_M_sol':r"$M_{\mathrm{crit}}$",
       'critical_rho0':r"$\rho_{\mathrm{crit}}$",
       'critical_r_c':r"$r_{\mathrm{crit}}$",
       'critical_beta':r"$\beta_{\mathrm{crit}}$",
       'Lbox':r"$L_{box}$",
       'N':r"$N$",
       'dx':r"$\Delta x$",
       'phase':r"$\mathrm{arg}(\Psi)$",
       'rho':r"$\rho$",
       'rhobar':r"$\bar{rho}$",
       'rho0':r"$\rho_0$",
       'i0':r"$i_0$",
       'beta':r"$\beta$",
       'r_c':r"$r_c$",
       'M_sol':r"$M_{\mathrm{sol}}$",
       'critical_f':r"$f_{\mathrm{crit}}$",
       'tailindex':r"$n_{\mathrm{tail}}$",
       'V':r"$V$",
       'v':r"$v$",
       'v2':r"$v^2$",
       'M':r"$M$",
       'W':r"$W$",
       'Kv':r"$K_v$",
       'Krho':r"$K_{\rho}$",
       'KQ':r"$K_Q$",
       'E':r"$E$",
       'L':r"$L$",
       'L2':r"$L^2$"}



## Constants



hbar = np.float64(1.71818131e-87)                                              # hbar / (mass of sun * (km/s) * kpc)
G = np.float64(4.3022682e-6)                                                   # G/((km/s)^2*kpc/mass of sun)
c = np.float64(299792.458)                                                     # c / (km/s)
pi = np.pi



## Main code (visualization and analysis)



## Conveniently extracts and calculates data from all snapshots in a folder.
## Attributed functions pertain to entire simulation as opposed to one frame.
## Snapshots stored as a list of Snap objects (below) as sim.snaps.
## Snap parameter loadall is toggled false, but data accumulates as you need it.
class Sim():
    
    def __init__(self, snapdir):
        
        self.snapdir = snapdir
        self.dir = os.path.basename(snapdir)
        
        self.snaps = [Snap(snapdir, i) for i in range(len(list(os.scandir(snapdir))))]
        self.t = np.array([s.t for s in self.snaps])
        self.Nout = len(self.snaps)
        
        s = self.snaps[0]
        self.m22 = s.m22
        self.m = s.m
        self.f15 = s.f15
        self.f = s.f
        self.a_s = s.a_s
        self.Lbox = s.Lbox
        self.N = s.N
        self.dx = s.dx
        
        self.critical_M_sol = s.critical_M_sol
        self.critical_rho0 = s.critical_rho0
        self.critical_r_c = s.critical_r_c
        self.critical_beta = s.critical_beta
        
        return
    
    
    
    ## Retrieves a quantity (along index kwargs) from any snaps in the simulation.
    def get(self, q, snaps=None, axis=None, project=False, i=None, iSlice=None, log10=False):
        
        if snaps is None: snaps = np.arange(0,len(self.snaps))
        
        data = []
        for j in range(len(snaps)):
            
            if j%(int(len(snaps)/4))==0: print(f"Retrieving {Q[q]} from Snap {snaps[j]}...")
            
            s = self.snaps[snaps[j]]
            data.append(s.get(q, axis, project, i, iSlice, log10))
        
        return data
    
    
    
    ## Plots the evolution of any scalar-valued quantities.
    def evolutionPlot(self, q, i=None, log10=False, ax=None, **kwargs):
        
        figsize = kwargs.pop('figsize', (9,3))
        dpi = kwargs.pop('dpi', 200)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        c = kwargs.pop('c', None)
        iterproduct = kwargs.pop('iterproduct', False)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps)))
        legendkws = kwargs.pop('legendkws', {'fontsize':'small'})
        
        combos = combineArguments(iterproduct, q, i=i, log10=log10, c=c)
        qs = [combo[0] for combo in combos]
        
        if ax is None:
            plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
        
        data = np.zeros((len(combos), len(snaps)))
        t = [self.t[s] for s in snaps]
        for j in range(len(combos)):
            
            print(f"Plotting quantity {j+1}: {Q[qs[j]]}...")
            
            q, i, log10, c = combos[j]
            data[j] = self.get(q, snaps=snaps, i=i, log10=log10)
            ax.plot(t, data[j], c=c, label=Q[q]+(i is not None)*f" {i}", **kwargs)
        
        ylabel = kwargs.pop('ylabel', ("Multiple Quantities" if len(combos)>1 else Q[q] + (i is not None)*f" {i}" + f" {U.get(U.get(q), '')}"))
        ax.set(xlabel=f"Time {U['time']}", ylabel=ylabel, title=r"($f_{15} = $" + rf"${self.f15}$)")
        plt.grid(True)
        plt.legend(**legendkws)    
        
        if save: 
            if filename is None:
                Qs = [Q[q] for q in qs]
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Evolution Plot"
            plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True) 
            
        return t, data
    
    
    
    ## Animates the evolution of any quantities defined throughout the box.
    def evolutionMovie(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        wspace = kwargs.pop('wspace', 0.3)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        fps = kwargs.pop('fps', 20)
        iterproduct = kwargs.pop('iterproduct', False)
        clims = kwargs.pop('clims', [None,None])
        climfactors = kwargs.pop('climfactors', [0,1])
        cmap = kwargs.pop('cmap', None)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps)))
        
        combos = combineArguments(iterproduct, q, axis, project, i, iSlice, log10, climfactors, clims, cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        qs = [combo[0] for combo in combos]
        
        clims = []
        for c in range(len(combos)):
            
            print(f"Computing color limits for plot {c+1}...")
            
            if combos[c][-2]==[None,None] or combos[c][-2]==[]:
                _q, _axis, _project, _i, _iSlice, _log10, climfactors, _1, _2 = combos[c]
                data = self.get(_q, snaps=snaps, axis=_axis, project=_project, i=_i, iSlice=_iSlice, log10=_log10)
                clims.append(getColorLimits(data, climfactors))
            else:
                clims.append(combos[c][-2])
        
        fig, ax = plt.subplots(1, len(combos), figsize=kwargs.pop('figsize', (4*len(combos),4)), dpi=dpi)
        if not hasattr(ax, '__len__'): ax = [ax]
        fig.subplots_adjust(wspace=wspace)
        colorbarmade = False
        
        def animate(j):
            
            if j%10==0: print(f"Animating frame {j+1}/{len(snaps)}")
            
            nonlocal colorbarmade
            
            [ax[k].clear() for k in range(len(ax))]
            
            s = snaps[j]
            self.snaps[s].plot2d(q, axis, project, i, iSlice, log10, ax, colorbar=(not colorbarmade), clims=clims, **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(snaps))
        writer = ani.PillowWriter(fps=fps)
        
        if save: 
            if filename is None:
                Qs = [Q[q] for q in qs]
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Evolution Plot"
            anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
        
        return
    
    
    
    ## Animates the evolution of the soliton density and its radial profile.
    def densityProfileMovie(self, rmin=None, rmax=None, shells=20, normalize=True, neighbors=1, rands=1e5, fit=False, **kwargs):
        
        figsize = kwargs.pop('figsize', (8,4))
        dpi = kwargs.pop('dpi', 200)
        clims = kwargs.pop('clims', [5,10])
        climfactors = kwargs.pop('climfactors', [0,1])
        axis = kwargs.pop('axis', 1)
        filename = kwargs.pop('filename', "Density Profile Movie")
        save = kwargs.pop('save', True)
        fps = kwargs.pop('fps', 10)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps)))
        
        if clims is None: 
            print("Computing color limits for density animation...")
            clims = getColorLimits(self.get('rho', snaps=snaps), climfactors)
        
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(wspace=0.3)
        colorbarmade = False
        
        def animate(i):
            
            if i%10==0: print(f"Animating frame {i+1}/{len(snaps)}")
        
            nonlocal colorbarmade
            
            ax[0].clear()
            ax[1].clear()
            
            s = snaps[i]
            self.snaps[s].slicePlot('rho', axis, iSlice='max', ax=ax[0], log10=True, colorbar=(not colorbarmade), clims=clims, **kwargs)
            self.snaps[s].densityProfile(rmin, rmax, shells, normalize, neighbors, rands, fit, False, ax[1], **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(snaps))
        writer = ani.PillowWriter(fps=fps)
        if save: anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
            
        return
    
    
    
    ## Retrieves folder path based on saving preferences and organizes given filename + extension.
    def getPathAndName(self, filename, ext):
        
        path = os.path.join(saveToParent, self.dir + " Images")
        if not os.path.isdir(path): path = saveToParent
        path = os.path.join(path, self.dir + " - " + filename + ext)
        
        return path



## Conveniently extracts and calculates data from one snapshot.
## Attributed functions pertain to one frame, not the whole simulation.
## Analogous to Philip's READSNAP.m helper function.
## Toggle loadall to choose how much data to extract from snapshot initially.
## get(): existence of attribute is checked first, then calculated and stored 
##        if necessary.
class Snap():
    
    ## Initializes Snap object.
    ## NOTE: Attributes within loadall clause cannot be referenced unless loadall
    ##       is True at instantiation or until attribute is retrived via get().
    def __init__(self, snapdir, snapnum, loadall=False):
        
        self.snapdir = snapdir
        self.num = snapnum
        self.filename = "snap{snapnum:04d}.h5".format(snapnum=snapnum)          
        snappath = os.path.join(snapdir, self.filename)
        f = h5py.File(snappath, 'r')
        self.path = snappath
        self.dir = os.path.basename(snapdir)
        
        self.psi = np.array(f['psiRe']) + 1j*np.array(f['psiIm'])
        self.t = float(f['time'][0])
        self.m22 = float(f['m22'][0])
        self.m = self.m22 * 8.96215327e-89
        self.f15 = float(self.dir[self.dir.find('f')+1:self.dir.find('L')])
        self.f = self.f15 * 8.05478166e-32
        self.a_s = hbar*c**3*self.m/(32*pi*self.f**2)                         
        self.Lbox = float(f['Lbox'][0])
        self.N = np.shape(self.psi)[0]
        self.dx = self.Lbox/self.N
        
        self.critical_M_sol = (hbar/np.sqrt(G*self.m*self.a_s) if self.a_s != 0 else np.inf)
        self.critical_rho0 = 1.2e9*self.m22**2*self.f15**4
        self.critical_r_c = 0.18/(self.m22*self.f15)
        self.critical_beta = 0.3
        
        self.all_loaded = loadall
        if loadall:
            self.phase = np.angle(self.psi)
            self.rho = np.abs(self.psi)**2
            self.rhobar = np.mean(self.rho)
            self.rho0 = np.max(self.rho)
            self.i0 = np.array(np.unravel_index(np.argmax(self.rho), self.rho.shape))
            self.beta = 1.6e-12/(self.m22)*self.rho0**(1/2) * hbar*c**5/(32*pi*G*self.f**2)
            self.r_c = (self.rho0/3.1e6)**(-1/4)*(2.5/self.m22)**(1/2)
            self.M_sol = 11.6*self.rho0*self.r_c**3
            self.critical_f = (self.rho0/1.2e9)**(1/4) * self.m22**(-1/2)
            self.tailindex = self.fitProfileTail(plot=False)[0][1]
            
            ## Get the potential.
            k = (2*pi/self.Lbox) * np.arange(-self.N/2, self.N/2)
            kx, ky, kz = np.meshgrid(k, k, k)
            kSq = np.fft.fftshift(kx**2 + ky**2 + kz**2)
            Vhat = -np.fft.fftn(4*pi*G*(self.rho - self.rhobar))/(kSq + (kSq==0))
            V = np.fft.ifftn(Vhat)
            self.V = np.real(V - np.mean(V))
            
            ## Get velocities. (v = nabla(S/m) where Psi = sqrt(rho)(e^(i*S/hbar)))
            v = gradient(self.phase, self.dx)*(2*self.dx)
            v[v > pi] = v[v > pi] - 2*pi
            v[v<=-pi] = v[v<=-pi] + 2*pi
            self.v = v/(2*self.dx)*(hbar/self.m)
            self.v2 = np.sum(self.v**2, axis=0)
            
            ## Get energies.
            self.M = np.sum(self.rho) * self.dx**3
            self.W = np.sum(self.rho*self.V/2) * self.dx**3
            self.Kv = np.sum(self.rho*(self.v[0]**2 + self.v[1]**2 + self.v[2]**2)/2) * self.dx**3
            dsqrtrho = gradient(np.sqrt(self.rho), self.dx)
            self.Krho = hbar**2/(2*self.m**2) * np.sum(dsqrtrho[0]**2 + dsqrtrho[1]**2 + dsqrtrho[2]**2) * self.dx**3
            dpsi = gradient(self.psi, self.dx)
            self.KQ = hbar**2/(2*self.m**2) * np.sum(np.abs(dpsi[0])**2 + np.abs(dpsi[1])**2 + np.abs(dpsi[2])**2) * self.dx**3
            self.E = self.W + self.Kv + self.Krho
            
            ## Angular momentum.
            x = (np.arange(self.N) + 1/2)*self.dx - self.Lbox/2
            x, y, z = np.meshgrid(x, x, x)
            Lx = np.sum(self.rho * (self.v[2]*y - self.v[1]*z)) * self.dx**3
            Ly = np.sum(self.rho * (self.v[0]*z - self.v[2]*x)) * self.dx**3
            Lz = np.sum(self.rho * (self.v[1]*x - self.v[0]*y)) * self.dx**3
            self.L = np.array([Lx, Ly, Lz])
            self.L2 = np.sum(self.L**2, axis=0)
        
        return
    
    
    
    ## Retrieves a quantity (along index kwargs) from the parent snap. 
    def get(self, q, axis=None, project=False, i=None, iSlice=None, log10=False):
        
        data = None
        
        if isinstance(i, int): i = [i]
        if isinstance(axis, str): axis = "xyz".find(axis)
        
        index = []
        full = (q in Q0)
        if q in Q1:
            if i is not None: 
                index = tuple(i)
            else: full = True
        elif q in Q3:
            if i is not None: 
                index = tuple(i)
            elif axis is not None and not project and iSlice != 'max':
                if iSlice is None: iSlice = int(self.N/2)-1
                index = tuple(np.roll([iSlice, slice(None), slice(None)], axis))
            elif iSlice is not None and iSlice != 'max':
                index = tuple([slice(None), iSlice, slice(None)])
            else:
                full = True
        elif q in Q4:
            if i is not None:
                index = tuple(i)
            else:
                index = tuple([slice(None)])
            if axis is not None and not project:
                if iSlice is None: iSlice = int(self.N/2)-1
                index = tuple(np.append(list(index), np.roll([iSlice, slice(None), slice(None)], axis)))
            elif iSlice is not None:
                index = tuple(np.append(list(index), [slice(None), iSlice, slice(None)]))
            if index==tuple([slice(None)]):
                full = True
        
        ## Gather all data
        
        if q=='snapdir': data = self.snapdir
        if q=='num': data = self.num
        if q=='filename': data = self.filename
        if q=='path': data = self.path
        if q=='dir': data = self.dir
        if q=='psi': data = self.psi
        if q=='t': data = self.t
        if q=='m22': data = self.m22
        if q=='m': data = self.m
        if q=='f15': data = self.f15
        if q=='f': data = self.f
        if q=='a_s': data = self.a_s                         
        if q=='critical_M_sol': data = self.critical_M_sol
        if q=='critical_rho0': data = self.critical_rho0
        if q=='critical_r_c': data = self.critical_r_c
        if q=='critical_beta': data = self.critical_beta
        if q=='Lbox': data = self.Lbox
        if q=='N': data = self.N
        if q=='dx': data = self.dx
        
        if q=='phase':
            try: data = self.phase if full else self.phase[index]
            except:
                if full:
                    self.phase = np.angle(self.psi)
                    data = self.phase
                else:
                    psi = self.psi[index]
                    data = np.angle(psi)
        if q=='rho': 
            try: data = self.rho if full else self.rho[index]
            except: 
                if full:
                    self.rho = np.abs(self.psi)**2
                    data = self.rho
                else:
                    psi = self.psi[index]
                    data = np.abs(psi)**2
        if q=='rhobar': 
            try: data = self.rhobar 
            except: 
                rho = self.get('rho')
                self.rhobar = np.mean(rho)
                data = self.rhobar
        if q in ['rho0', 'i0']:
            try: data = self.rho0 if q=='rho0' else self.i0
            except:
                rho = self.get('rho')
                self.i0 = np.unravel_index(np.argmax(rho), rho.shape)
                self.rho0 = rho[self.i0]
                data = self.rho0 if q=='rho0' else self.i0
        if q=='beta':
            try: data = self.beta
            except:
                rho0 = self.get('rho0')
                self.beta = 1.6e-12/(self.m22)*rho0**(1/2) * hbar*c**5/(32*pi*G*self.f**2)
                data = self.beta
        if q=='r_c':
            try: data = self.r_c
            except:
                rho0 = self.get('rho0')
                self.r_c = (rho0/3.1e6)**(-1/4)*(2.5/self.m22)**(1/2)
                data = self.r_c
        if q=='M_sol':
            try: data = self.M_sol
            except:
                rho0 = self.get('rho0')
                r_c = self.get('r_c')
                self.M_sol = 11.6*rho0*r_c**3
                data = self.M_sol
        if q=='critical_f':
            try: data = self.critical_f
            except:
                rho0 = self.get('rho0')
                self.critical_f = (rho0/1.2e9)**(1/4) * self.m22**(-1/2)
                data = self.critical_f
        if q=='tailindex':
            try: data = self.tailindex
            except:
                self.tailindex = self.fitProfileTail(plot=False)[0][1]
                data = self.tailindex
        if q=='V':
            try: data = self.V if full else self.V[index]
            except:
                rho = self.get('rho')
                rhobar = self.get('rhobar')
                k = (2*pi/self.Lbox) * np.arange(-self.N/2, self.N/2)
                kx, ky, kz = np.meshgrid(k, k, k)
                kSq = np.fft.fftshift(kx**2 + ky**2 + kz**2)
                Vhat = -np.fft.fftn(4*pi*G*(rho - rhobar))/(kSq + (kSq==0))
                V = np.fft.ifftn(Vhat)
                self.V = np.real(V - np.mean(V))
                data = self.V if full else self.V[index]
        if q=='v':
            try: data = self.v if full else self.v[index]
            except:
                phase = self.get('phase')
                v = gradient(phase, self.dx)*(2*self.dx)
                v[v > pi] = v[v > pi] - 2*pi
                v[v<=-pi] = v[v<=-pi] + 2*pi
                self.v = v/(2*self.dx)*(hbar/self.m)
                data = self.v if full else self.v[index]
        if q=='v2':
            try: data = self.v2 if full else self.v2[index]
            except:
                v = self.get('v')
                self.v2 = np.sum(v**2, axis=0)
                data = self.v2 if full else self.v2[index]
        if q=='M': 
            try: data = self.M
            except:
                rho = self.get('rho')
                self.M = np.sum(rho) * self.dx**3
                data = self.M
        if q=='W': 
            try: data = self.W
            except:
                rho = self.get('rho')
                V = self.get('V')
                self.W = np.sum(rho*V/2) * self.dx**3
                data = self.W
        if q=='Kv': 
            try: data = self.Kv
            except:
                rho = self.get('rho')
                v2 = self.get('v2')
                self.Kv = np.sum(rho*v2/2) * self.dx**3
                data = self.Kv
        if q=='Krho':
            try: data = self.Krho
            except:
                rho = self.get('rho')
                dsqrtrho = gradient(np.sqrt(self.rho), self.dx)
                self.Krho = hbar**2/(2*self.m**2) * np.sum(dsqrtrho[0]**2 + dsqrtrho[1]**2 + dsqrtrho[2]**2) * self.dx**3
                data = self.Krho
        if q=='KQ':
            try: data = self.KQ
            except:
                dpsi = gradient(self.psi, self.dx)
                self.KQ = hbar**2/(2*self.m**2) * np.sum(np.abs(dpsi[0])**2 + np.abs(dpsi[1])**2 + np.abs(dpsi[2])**2) * self.dx**3
                data = self.KQ
        if q=='E':
            try: data = self.E
            except:
                W = self.get('W')
                Kv = self.get('Kv')
                Krho = self.get('Krho')
                KQ = self.get('KQ')
                self.E = W + Kv + Krho + KQ
                data = self.E
        if q=='L':
            try: data = self.L if full else self.L[index]
            except:
                rho = self.get('rho')
                v = self.get('v')
                x = (np.arange(self.N) + 1/2)*self.dx - self.Lbox/2
                x, y, z = np.meshgrid(x, x, x)
                Lx = np.sum(self.rho * (self.v[2]*y - self.v[1]*z)) * self.dx**3
                Ly = np.sum(self.rho * (self.v[0]*z - self.v[2]*x)) * self.dx**3
                Lz = np.sum(self.rho * (self.v[1]*x - self.v[0]*y)) * self.dx**3
                self.L = np.array([Lx, Ly, Lz])
                data = self.L if full else self.L[index]
        if q=='L2':
            try: data = self.L2
            except:
                L = self.get('L')
                self.L2 = np.sum(L**2)
                data = self.L2
        
        if project:
            if axis is None: axis = 1
            if q in Q3:
                data = getProjection(data, axis)
            if q in Q4:
                data = [getProjection(data_part, axis) for data_part in data]
        
        if iSlice == 'max':
            if axis is None: axis = 1
            if q in Q3:
                data = getSlice(data, axis, 'max')
            if q in Q4:
                data = [getSlice(data_part, axis, 'max') for data_part in data]
        
        if isinstance(i, str): 
            if isinstance(data, dict): 
                data = data[i]
            else:
                print("String index given for non-dictionary-type quantity.")
            
        
        if log10: data = np.log10(data)  
        
        return data
    
    
    
    ## Plots a single given quantity defined throughout the box.
    ## NOTE: This function is a helper. Use plot2d for same functionality +
    ##       multi-quantity support.
    def singlePlot2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        figsize = kwargs.pop('figsize', None)
        dpi = kwargs.pop('dpi', 200)
        zoom = kwargs.pop('zoom', [0,self.N,0,self.N])
        cmap = kwargs.pop('cmap', C.get(q, 'viridis'))
        if cmap is None: cmap = C.get(q, 'viridis')
        climfactors = kwargs.pop('climfactors', [0,1])
        colorbar = kwargs.pop('colorbar', True)
        title = kwargs.pop('title', None)
        
        if ax is None:
            plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
        
        full_data = self.get(q)
        data = self.get(q, axis, project, i, iSlice, log10)
        if isinstance(axis, str): axis = "xyz".find(axis)
        if iSlice is None: iSlice = int(self.N/2)-1
        if iSlice == 'max': iSlice = np.unravel_index(np.argmax(full_data), full_data.shape)[axis]
        if len(np.shape(data))!=2: print(f"Requested data is not 2-dimensional (shape {np.shape(data)})."); return
        
        data = data[zoom[0]:zoom[1], zoom[2]:zoom[3]]
        extent = np.array([zoom[2], zoom[3], zoom[1], zoom[0]])*self.dx
        
        clims = kwargs.pop('clims', getColorLimits(data, climfactors))
        im = ax.imshow(data, extent=extent, cmap=cmap, vmin=clims[0], vmax=clims[1], **kwargs)
        axes = ['yz', 'zx', 'xy'][axis]
        ax.set(xlabel=rf"${axes[0]}$ [kpc]", ylabel=rf"${axes[1]}$ [kpc]")
        if title is None: title = r"($f_{15}$" + rf"$ = {self.f15}$, " + ("" if project else rf"${'xyz'[axis]} = {np.round((iSlice+0.5)*self.dx,2)}$, ") + rf"$t = {np.round(self.t,3)}$)"
        ax.set_title(title)
        
        if colorbar is True:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(log10*r'$\log_{10}($' + Q[q] + log10*')' + f" {U.get(U.get(q), '')}")
        
        return data
    
    
    
    ## Plots multiple quantities defined throughout the box.
    ## Same input parameters as Snap.singlePlot2d, but you can replace any parameter with a list.
    def plot2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        clims = kwargs.pop('clims', [None,None])
        zoom = kwargs.pop('zoom', [0,self.N,0,self.N])
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', None)
        wspace = kwargs.pop('wspace', 0.3)
        iterproduct = kwargs.pop('iterproduct', False)
        cmap = kwargs.pop('cmap', None)
        
        combos = combineArguments(iterproduct, q, axis, project, i, iSlice, log10, climfactors, clims, cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        
        qs = [combo[0] for combo in combos]
        
        if ax is None:
            fig, ax = plt.subplots(1, len(combos), figsize=kwargs.pop('figsize', (4*len(combos), 4)), dpi=dpi)
            fig.subplots_adjust(wspace=wspace)
        if not hasattr(ax, '__len__'): ax = np.array([ax])
        
        data = np.zeros((len(combos), zoom[1]-zoom[0], zoom[3]-zoom[2]))
        for j in range(len(combos)):
            q, axis, project, i, iSlice, log10, climfactors, clims, cmap = combos[j]
            data[j] = self.singlePlot2d(q, axis, project, i, iSlice, log10, climfactors=climfactors, clims=clims, cmap=cmap, ax=ax[j], zoom=zoom, **kwargs)
        
        if save: 
            if filename is None:
                Qs = [Q[q] for q in qs]
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Evolution Plot"
            plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return data
    
    
    
    ## Plots a cross-section of any quantities defined throughout the box.
    def slicePlot(self, q, axis=1, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        return self.plot2d(q, axis=axis, i=i, iSlice=iSlice, log10=log10, ax=ax, **kwargs)

        
    
    ## Plots a projection of a quantity defined throughout the box.
    ## By "projection", I mean "maximum along an axis" for now (no weighting).
    def projectionPlot(self, q, axis=1, i=None, log10=False, ax=None, **kwargs):
        
        return self.plot2d(q, axis=axis, project=True, i=i, log10=log10, ax=ax, **kwargs)
    
    
    
    ## Animates cross-sections of quantities through the box.
    def scan3d(self, q, axis=1, i=None, log10=False, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        clims = kwargs.pop('clims', [None,None])
        cmap = kwargs.pop('cmap', None)
        wspace = kwargs.pop('wspace', 0.3)
        fps = kwargs.pop('fps', 8)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        iterproduct = kwargs.pop('iterproduct', False)
        frames = kwargs.pop('frames', np.arange(0, self.N))
        
        combos = combineArguments(iterproduct, q, axis, '/', i, '/', log10, climfactors, clims, cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        qs = [combo[0] for combo in combos]
        
        clims, data = [], None
        for c in range(len(combos)):
            
            print(f"Computing color limits for plot {c+1}...")
            
            if combos[c][-2]==[None,None] or combos[c][-2]==[]:
                _q, _axis, _i, _log10, climfactors, _1, _2 = combos[c]
                data = self.get(_q, i=_i, log10=_log10)
                clims.append(getColorLimits(data, climfactors))
            else:
                clims.append(combos[c][-2])
        
        fig, ax = plt.subplots(1, len(combos), figsize=(4*len(combos),4), dpi=dpi)
        fig.subplots_adjust(wspace=wspace)
        if not hasattr(ax, '__len__'): ax = [ax]
        colorbarmade = False
        
        def animate(j):
            
            if j%10==0: print(f"Animating frame {j+1}/{len(frames)}")
            
            nonlocal colorbarmade
            
            [a.clear() for a in ax]
            
            iSlice = frames[j]
            self.plot2d(q, axis, i=i, iSlice=iSlice, log10=log10, ax=ax, clims=clims, colorbar=(not colorbarmade), **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(frames))
        writer = ani.PillowWriter(fps=fps)
        if save: 
            if filename is None:
                Qs = [Q[q] for q in qs]
                if len(Qs)==1: Qs = Qs[0]
            anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
        
        return data
    
    
    
    ## Computes the soliton density profile.
    ## Toggle 'plot' to plot and 'fit' to include the theoretical profile.
    def densityProfile(self, rmin=None, rmax=None, shells=20, normalize=True, neighbors=1, rands=1e5, fit=False, plot=True, ax=None, **kwargs):
        
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', "Density Profile")
        
        rho, rho0, r_c, iB = self.get('rho'), self.get('rho0'), self.get('r_c'), self.get('i0')
        
        i0 = np.array([int(self.N/2)-1]*3)
        rho = np.roll(rho, list(i0-iB), axis=np.arange(3))
        
        x_ = (np.arange(100)+0.5)*self.dx
        x, y, z = np.meshgrid(x_, x_, x_)
        x0, y0, z0 = x_[i0]
        
        iN = slice(i0[0]-neighbors, i0[0]+neighbors+1)
        xN = x_[(iN)]
        mN = np.ravel(rho[iN, iN, iN]*self.dx**3)
        coords = np.array([np.ravel(x) for x in np.meshgrid(xN, xN, xN)])
        xCM, yCM, zCM = np.sum(mN*coords, axis=1)/np.sum(mN)
        offset = np.sqrt((x0-xCM)**2 + (y0-yCM)**2 + (z0-zCM)**2)
        
        if rmin is None: rmin = offset/2
        elif normalize: rmin = rmin*r_c
        if rmax is None: rmax = self.Lbox/2
        elif normalize: rmax = rmax*r_c
        
        r = np.logspace(np.log10(rmin), np.log10(rmax), shells)
        mids = 10**np.array([np.mean(np.log10([r[i], r[i+1]])) for i in range(shells-1)])
        
        rho_r = np.zeros(shells-1)
        for i in range(shells-1):
            random_coords = np.random.uniform(-1,1,size=(3, int(rands)))
            norm = np.sqrt(np.sum(random_coords**2, axis=0))
            random_coords = random_coords/norm
            r_for_randoms = np.logspace(np.log10(r[i]), np.log10(r[i+1]), int(rands))
            random_coords = random_coords * r_for_randoms
            random_coords = np.array([random_coords[0]+xCM, random_coords[1]+yCM, random_coords[2]+zCM])
            random_i = np.int32(np.round(random_coords/self.dx-0.5))
            rho_r[i] = np.mean(rho[random_i[0], random_i[1], random_i[2]])
        
        if normalize: rho_r, mids = rho_r/rho0, mids/r_c
        
        if plot or save or ax is not None:
            
            figsize = kwargs.pop('figsize', (6,4))
            dpi = kwargs.pop('dpi', 200)
            lims = kwargs.pop('lims', ([1e-1,1e2,1e-5,1] if normalize else [5e-2,5e1,1e2,1e10]))
            
            if ax is None: fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            label = rf'$\rho_0 = {np.round(rho0/1e10,1)}$' + r' [$10^{10}$ ' + U['density'] + ']'
            ax.loglog(mids, rho_r, 'o-', label=label, **kwargs)
            ax.set_xlabel(r"$r$" + normalize*r"$/r_c$" + (not normalize)*f" {U['length']}")
            ax.set_ylabel(r"$\rho(r)$" + normalize*r"$/\rho_0$" + (not normalize)*f" {U['density']}")
            ax.set_title(r"($f_{15}$" + rf"$ = {self.f15}$, $t = {np.round(self.t,3)}$)")
            ax.set(xlim=(lims[0], lims[1]), ylim=(lims[2], lims[3]))
            
            if fit:
                r_fit = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 100)
                rho_fit_noSI = self.solitonDensityFunction(r_fit, normalize=normalize, noSI=True)
                ax.loglog(r_fit, rho_fit_noSI, c='k', label='Theoretical Non-SI Profile')
                if self.a_s != 0.0:
                    rho_fit = self.solitonDensityFunction(r_fit, normalize=normalize)
                    ax.loglog(r_fit, rho_fit, c='gray', label='Theoretical SI Profile (Not Collapsed)')
                plt.vlines((self.dx/r_c if normalize else self.dx), lims[2], lims[3], linestyles='dashed', label="Length of One Grid Cell", color="gray")
                plt.vlines((3.5 if normalize else 3.5*r_c), lims[2], lims[3], linestyles='solid', label="Theoretical Non-SI Universal Cutoff", color="rosybrown")
            
            ax.grid()
            plt.legend(loc=3, fontsize='small')
            
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return mids, rho_r
    
    
    
    ## Theoretical soliton density function of radius (before collapse).
    def solitonDensityFunction(self, r, normalize=True, **kwargs):
        
        noSI = kwargs.pop('noSI', False)
        rho0 = kwargs.pop('rho0', self.get('rho0'))
        r_c = kwargs.pop('r_c', self.get('r_c'))
        beta = kwargs.pop('beta', (0 if noSI else self.get('beta')))
        
        def i1(b): return np.tanh(b/5)
        def i2(b): return np.tanh(b)
        def i3(b): return np.tanh(np.sqrt(b))**2
        
        if not normalize: r = r/r_c
        
        rho = (1 + (1+2.60*i1(beta)) * 0.091 * (r*np.sqrt(1+beta))**(2-i2(beta)/5))**(-8+22/5*i3(beta))
        
        if not normalize: rho = rho*rho0
        
        return rho
    
    
    
    ## Fits tail of soliton density profile to a power law.
    def fitProfileTail(self, rmin=7.0, plot=True, ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', "Density Profile Tail")
        
        Lbox, r_c = self.get('Lbox'), self.get('r_c')
        
        def powerLawLine(r, A, n): return A + n*r
        
        r, rho = self.densityProfile(rmin, Lbox/2/r_c, normalize=True, plot=False)
        fit = curve_fit(powerLawLine, np.log10(r), np.log10(rho), p0=[-3,-3])
        
        if plot:
            if ax is None: fig, ax = plt.subplots(dpi=dpi)
            ax.loglog(r, rho, 'o-', label="Measured Density", **kwargs)
            
            A_fit, n_fit = fit[0]
            r_fit = np.linspace(rmin, Lbox/2/r_c, 100)
            rho_fit = 10**(powerLawLine(np.log10(r_fit), A_fit, n_fit))
            ax.loglog(r_fit, rho_fit, 'k', label=f"$n = {np.round(n_fit,4)}$")
        
            ax.set_xlabel(r"$r/r_c$")
            ax.set_ylabel(r"$\rho(r)/\rho_0$")
            ax.set_title(r"($f_{15}$" + rf"$ = {self.f15}$, $t = {np.round(self.t,3)}$)")
            ax.grid()
            plt.legend()
        
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)    
        
        return fit
    
    
    
    ## Retrieves folder path based on saving preferences and organizes given filename + extension.
    def getPathAndName(self, filename, ext):
        
        path = os.path.join(saveToParent, self.dir + " Images")
        if not os.path.isdir(path): path = saveToParent
        append = self.dir + " - " + str(self.num) + " - " + filename + ext
        path = os.path.join(path, append)
        
        return path



## Helper functions



## Gets the gradient of a quantity defined in the box.
def gradient(f, dx, i=[0,1,2]): 
    
    axes = np.arange(3)
    grad = []
    if 0 in i: grad.append((np.roll(f, [0,-1,0], axis=axes) - np.roll(f, [0,1,0], axis=axes))/(2*dx))
    if 1 in i: grad.append((np.roll(f, [-1,0,0], axis=axes) - np.roll(f, [1,0,0], axis=axes))/(2*dx))
    if 2 in i: grad.append((np.roll(f, [0,0,-1], axis=axes) - np.roll(f, [0,0,1], axis=axes))/(2*dx))
    
    return np.array(grad)



## Given 3-dimensional data, returns a 2-d cross-section along an axis.
## If sheet is None, returns slice containing largest value in data.
def getSlice(data, axis, iSlice='max'):
    
    if isinstance(axis, str): axis = "xyz".find(axis)
    
    if iSlice=='max': iSlice = np.unravel_index(np.argmax(data), data.shape)[axis]
    if axis==0: data = data[iSlice,:,:]
    if axis==1: data = data[:,iSlice,:]
    if axis==2: data = data[:,:,iSlice]
    
    return data



## Given 3-dimensional data, returns a 2-d projection along an axis.
def getProjection(data, axis):
    
    if isinstance(axis, str): axis = "xyz".find(axis)
    
    return np.max(data, axis=axis)



## Given a set of numerical data, computes the color limits for colored plots.
## If factors are not supplied, limits default to min and max of data.
## 'factors' span the range between min and max of data.
def getColorLimits(data, factors=(0,1)):
    
    low, high = np.min(data), np.max(data)
    clims = (low + factors[0]*(high-low), high - (1-factors[1])*(high-low))
    
    return clims



## Creates subfolders to organize saved images within given parent directory.
def createImageSubfolders(parent=os.getcwd()):
    
    subfolders = []
    for folder in folders:
        dirname = os.path.join(parent, os.path.basename(folder) + " Images")
        subfolders.append(dirname)
        if not os.path.isdir(dirname): 
            os.mkdir(dirname)
            
    return subfolders



## Given ragged assortment of arguments, returns coherent lists of arguments
## for execution in multi-quantity plotting functions.
def combineArguments(iterproduct=False, q='/', axis='/', project='/', i='/', iSlice='/', log10='/',
                     climfactors='/', clims='/', cmap='/', c='/'):
    
    if isinstance(q, str) and q!='/': q = [q]
    if (isinstance(axis, int) or axis is None) and axis!='/': axis = [axis]
    if isinstance(project, bool) and project!='/': project = [project]
    if isinstance(i, list):
        if not np.any([hasattr(i[j], '__len__') for j in range(len(i))]): i = [i]
    elif i!='/': i = [i]
    if (not isinstance(iSlice, list)) and iSlice!='/': iSlice = [iSlice]
    if isinstance(log10, bool) and log10!='/': log10 = [log10]
    if len(np.shape(climfactors))<2 and climfactors!='/': climfactors = [climfactors]
    if len(np.shape(clims))<2 and clims!='/': clims = [clims]
    if (isinstance(cmap, str) or cmap is None) and cmap!='/': cmap = [cmap]
    if (isinstance(c, str) or c is None) and c!='/': c = [c]
    
    l = {'q':q, 'axis':axis, 'project':project, 'i':i, 'iSlice':iSlice, 'log10':log10, 'climfactors':climfactors, 'clims':clims, 'cmap':cmap, 'c':c}
    args = [l[j] for j in l if l[j]!='/']
    
    combos = []
    if iterproduct:
        combos = list(itertools.product(*args))
    else:
        maxLen = np.max([len(arg) for arg in args])
        for i in range(maxLen):
            combo = [(arg[i] if len(arg)==maxLen else arg[0]) for arg in args]
            combos.append(combo)
    
    return combos



## Catalog of conversion factors from code units to other units.
def unitsFactor(u, to='SI'):
    
    to = to.lower()
    factor = 1
    if u in ['length', 'l']: 
        if to=='si': factor = 3.086e19
        if to=='km': factor = 3.086e16
        if to=='cm' or to=='esu': factor = 3.086e22
    
    if u in ['mass', 'm']: 
        if to=='si': factor = 1.989e30
        if to=='g' or to=='esu': factor = 1.989e33
    
    if u in ['time', 't']: 
        if to=='si': factor = 3.086e16
        if to=='yr': factor = 9.786e8
        if to=='myr': factor = 9.786e2
        if to=='gyr': factor = 9.786e-1
    
    if u in ['energy', 'e']: 
        if to=='si': factor = 1.989e36
        if to=='gev': factor = 1.242e46
        if to=='esu': factor = 1.989e43
    
    if u in ['velocity', 'v']: 
        if to=='si': factor = 1e3
        if to=='esu': factor = 1e5
    
    if u in ['density']: 
        if to=='si': factor = 6.768e-29
        if to=='esu': factor = 6.768e-32
    
    if factor==1: raise KeyError("Support for requested unit is not yet implemented.")
    
    return factor





















## Handles folder checks and subfolder creation.
if __name__=='__main__':
    
    global imageSubfolders
    okay = []
    goodfolders = []
    for f in folders:
        okay.append(os.path.isdir(f))
        if okay[-1]:
            goodfolders.append(f)
        else:
            print(f"Folder {f} was not found.")
    folders = goodfolders
    
    if createImgSubfolders:
        imageSubfolders = createImageSubfolders(parent=saveToParent)
    
        
        
        
        