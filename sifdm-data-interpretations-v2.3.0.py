"""
SIFDM Data Interpretations

- Created by Connor Painter on 9/27/21 to do elementary analysis on the outputs of SIFDM MATLAB code by Philip Mocz.
- Last updated: 02/14/22
"""



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import h5py
import os
import time
import itertools
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.integrate import quad
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'



"""
GETTING STARTED 

- Prepare to import your data (from Philip's .h5 output) by listing paths (strings) to output folders in FOLDER SETUP.
    -   List must be named 'folders'.
    -   Feel free to use my formatting below.
- Set saving preferences with 'createImgSubfolders' and 'saveToParent' below in PREFERENCES.
    -   'saveToParent': the parent directory to which you want images saved.
    -   'createImgSubfolders': create subfolders to organize saved images.
"""


"""
FOLDER SETUP

- Import your data (from Philip's .h5 output) here.
"""



## f15 = Inf.; L = 20; T = 16 Nout = 1600; N (res) = 100
dInf = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/fInfL20T16n1600r100"

## f15 = Inf.; L = 20; T = 4 Nout = 40; N (res) = 400
dInfHD = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/fInfL20T4n40r400"

## f15 = 4.00; L = 20; T = 16; Nout = 1600; N (res) = 100
d4 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f4L20T16n1600r100"

## f15 = 2.75; L = 20; T = 16; Nout = 10; N (res) = 100
d2_75 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f2.75L20T16n1600r100"

## f15 = 2.00; L = 20; T = 16; Nout = 1600; N (res) = 100
d2 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f2L20T16n1600r100"

## f15 = 2.00; L = 20; T = 4; Nout = 40; N (res) = 400
d2HD = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f2L20T4n40r400"

## f15 = 1.00; L = 20; T = 16; Nout = 1600; N (res) = 100
d1 = r"/Users/cap/Documents/UT/Research (Boylan-Kolchin)/SIFDM Project/sifdm-matlab-main/output/f1L20T16n1600r100"



folders = [dInf, d4, d2_75, d2, d1, dInfHD, d2HD]



"""
PREFERENCES

- Customize how you want your images saved here.
"""



createImgSubfolders = True
saveToParent = os.path.join(os.getcwd(), "Saved Figures")
onlyNameF15 = False



"""
USEFUL ARRAYS AND DICTIONARIES

- Q: given quantity codename, outputs a more detailed name
- Q0: lists quantities that are naturally described as scalars at a given time
- Q1: lists quantities that are naturally described as vectors at a given time
- Q2: lists quantities that are naturally described as matrices at a given time
- Q3: lists quantities that are naturally described as rank-3 tensors at a given time
- Q4: lists quantities that are naturally described as rank-4 tensors at a given time
- U: given physical unit description or dimensionful quantity codename, outputs associated code units
- C: given certain Q3 or Q4 quantity codename, outputs default colormap
- LTX: given quantity codename, outputs associated LaTeX to draw symbols
"""



Q = {'':'', 
     'snapdir':'Path to Snapshot-Enclosing Directory', 
     'num':'Snapshot Number', 'filename':'Snapshot Filename', 
     'path':'Path to Snapshot', 
     'dir':'Snapshot-Enclosing Directory', 
     'psi':'Wavefunction', 
     't':'Time', 
     'm22':'Normalized Particle Mass',
     'm':'Particle Mass', 
     'f15':'Normalized Self-interaction Parameter', 
     'f':'Self-interaction Parameter', 
     'a_s':'s-scattering Length', 
     'crit_M_sol':'Critical Soliton Mass', 
     'crit_rho0':'Critical Soliton Central Density', 
     'crit_r_c':'Critical Soliton Core Radius',
     'crit_beta':'Critical Beta',
     'Lbox':'Box Side Length', 
     'N':'Resolution', 
     'dx':'Grid Spacing',
     'phase':'Wavefunction Phase', 
     'rho':'Density', 
     'rhobar':'Mean Density', 
     'rhoMax':'Maximum Density',
     'iMax':'Maximum Density Index',
     'profile':'Density Profile',
     'fitdict':'Dictionary of Profile Parameter Fits',
     'rho0':'Soliton Central Density', 
     'delta':'Density Profile Delta Factor',
     'r_c':'Soliton Core Radius',
     'cutoff':'Soliton-Tail Density Cutoff',
     'M_sol':'Soliton Mass',
     'crit_f15':'Critical Self-interaction Parameter',
     'beta':'Beta (Soliton Stability Parameter)',
     'n':'Density Profile Tail Index',
     'n_tail':'Density Profile Tail Index Limit',
     'A_tail':'Density Profile Tail Amplitude Fit',
     'solitonGOF':'Goodness of Fit to Soliton Profile',
     'tailGOF':'Goodness of Fit to Density Profile Tail',
     'profileGOF':'Goodness of Fit to Entire Density Profile',
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
Q0 = ['t', 'm22', 'm', 'f15', 'f', 'a_s', 'crit_M_sol', 'crit_rho0', 'crit_r_c', 'crit_beta', 'Lbox', 'N', 'dx', 'rhobar', 'rho0', 'beta', 'r_c', 'M_sol', 'crit_f15', 'tailindex', 'M', 'W', 'Kv', 'Krho', 'KQ', 'L2']
Q1 = ['L', 'i0']
Q2 = []
Q3 = ['psi', 'phase', 'rho', 'V', 'v2']
Q4 = ['v']
U = {'length':'[kpc]', 'mass':r"[$M_{\odot}$]", 'energy':r"[$M_{\odot}\mathrm{(km/s)}^2$]", 'time':"[kpc/(km/s)]", 'density':r"[$M_{\odot}/\mathrm{kpc}^3$]", 'velocity':"[km/s]",
     't':'time', 'm':'mass', 'f':'energy', 'Lbox':'length', 'dx':'length', 'rho':'density', 'rhobar':'density', 'rho0':'density', 'r_c':'length', 'M_sol':'mass', 'E':'energy', 'W':'energy', 'Kv':'energy', 'Krho':'energy', 'KQ':'energy'}
C = {'phase':'bwr', 'rho':'inferno', 'V':'cividis', 'v':'hot', 'v2':'hot'}
LTX = {'psi':r"$\Psi$",
       't':r"$t$",
       'm22':r"$\frac{m}{10^{-22}\ \mathrm{eV}}$",
       'm':r"$m$",
       'f15':r"$\frac{f}{10^{15}\ \mathrm{eV}}$",
       'f':r"$f$",
       'a_s':r"$a_s$",
       'crit_M_sol':r"$M_{\mathrm{crit}}$",
       'crit_rho0':r"$\rho_{\mathrm{crit}}$",
       'crit_r_c':r"$r_{\mathrm{crit}}$",
       'crit_beta':r"$\beta_{\mathrm{crit}}$",
       'Lbox':r"$L_{box}$",
       'N':r"$N$",
       'dx':r"$\Delta x$",
       'phase':r"$\mathrm{arg}(\Psi)$",
       'rho':r"$\rho$",
       'rhobar':r"$\bar{\rho}$",
       'profile':r"$\rho(r)$",
       'rho0':r"$\rho_0$",
       'i0':r"$i_0$",
       'beta':r"$\beta$",
       'r_c':r"$r_c$",
       'M_sol':r"$M_{\mathrm{sol}}$",
       'crit_f15':r"$f_{\mathrm{crit}}$",
       'n_tail':r"$n_{\mathrm{tail}}$",
       'A_tail':r"$A_{\mathrm{tail}}$",
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



"""
CONSTANTS
"""



hbar = np.float64(1.71818131e-87)                                              # hbar / (mass of sun * (km/s) * kpc)
G = np.float64(4.3022682e-6)                                                   # G/((km/s)^2*kpc/mass of sun)
c = np.float64(299792.458)                                                     # c / (km/s)
pi = np.pi



"""
SIM AND SNAP OBJECTS

- Classes of objects representing single snapshots or entire simulations.
- Intuitively attributed functions for plotting, animation, and analysis.
"""



class Sim():
    
    """
    Extracts and calculates data from all snapshots in a folder.
    - snapdir: path to output directory
    - lite: do not load psi at initialization
    - store: store data in snaps when computed
    """
    
    def __init__(self, snapdir, lite=True, store=True):
        
        self.snapdir = snapdir
        self.dir = os.path.basename(snapdir)
        self.lite = lite
        self.store = store
        
        print(f"Loading {lite*'(lite) '}Snap objects from folder {self.dir}...")
        
        self.snaps = [Snap(snapdir, i, lite=lite) for i in range(len(list(os.scandir(snapdir))))]
        self.t = [s.t for s in self.snaps]
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
        
        self.crit_M_sol = s.crit_M_sol
        self.crit_rho0 = s.crit_rho0
        self.crit_r_c = s.crit_r_c
        self.crit_beta = s.crit_beta
        
        return
    
    
    
    def get(self, q, snaps=None, axis=None, project=False, i=None, iSlice=None, log10=False, **kwargs):
        
        """
        Retrieves a quantity (along index kwargs) from any snaps in the simulation.
        - q: name of quantity
        - snaps: list of snapshot numbers
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        """
        
        if snaps is None: snaps = np.arange(0,len(self.snaps))
        
        data = []
        for j in range(len(snaps)):
            
            if j%50==0: print("Retrieving {} from Snap {} ({:.2%})...".format(Q[q], snaps[j], j/len(snaps)))
            
            s = self.snaps[snaps[j]] if self.store else Snap(self.snapdir, snaps[j])
            data.append(s.get(q, axis, project, i, iSlice, log10, **kwargs))
        
        return np.array(data)
    
    
    
    def evolutionPlot(self, q, i=None, log10=False, ax=None, **kwargs):
        
        """
        Plots the evolution of any scalar-valued quantities.
        - q: name of quantity(s)
        - i: index of multi-dimensional quantity
        - log10: plot log (base 10) of quantity
        - ax: axes on which to plot
        """
        
        figsize = kwargs.pop('figsize', (9,3))
        dpi = kwargs.pop('dpi', 200)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        c = kwargs.pop('c', None)
        iterproduct = kwargs.pop('iterproduct', False)
        eo = kwargs.pop('eo', 1)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps),eo))
        legendkws = kwargs.pop('legendkws', {'fontsize':'small'})
        smooth = kwargs.pop('smooth', None)
        annotate = kwargs.pop('annotate', True)
        
        combos = combineArguments(iterproduct, q=q, i=i, c=c, smooth=smooth)
        Qs = list(set([Q[combo[0]] for combo in combos]))
        
        if ax is None: fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        data = np.zeros((len(combos), len(snaps)))
        t = [self.t[s] for s in snaps]
        for j in range(len(combos)):
            
            q, i, c, smooth = combos[j]
            
            if annotate: print(f"Plotting quantity {j+1}: {Q[q]}...")
            
            data[j] = self.get(q, snaps=snaps, i=i)
            if isinstance(smooth, int): data[j] = SMA(data[j], smooth)
            
            label = Q[q] + (i is not None)*f" {i}" + isinstance(smooth,int)*f" (SMA = {smooth})"
            ax.plot(t, data[j], c=c, label=label, **kwargs)
        
        ylabel = kwargs.pop('ylabel', ("Multiple Quantities" if len(combos)>1 else Q[q] + (i is not None)*f" {i}" + f" {U.get(U.get(q), '')}"))
        ax.set(xlabel=f"Time {U['time']}", ylabel=ylabel, title=r"($f_{15} = $" + rf"${self.f15}$)", yscale=['linear','log'][log10])
        plt.grid(True)
        plt.legend(**legendkws)    
        
        if save: 
            if filename is None:
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Evolution Plot"
            plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True) 
            
        return t, data
    
    
    
    def evolutionMovie(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, **kwargs):
        
        """
        Animates the evolution of any quantities defined throughout the box.
        - q: name of quantity(s)
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        """
        
        dpi = kwargs.pop('dpi', 200)
        wspace = kwargs.pop('wspace', 0.3)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        ext = kwargs.pop('ext', '.gif')
        fps = kwargs.pop('fps', 20)
        iterproduct = kwargs.pop('iterproduct', False)
        clims = kwargs.pop('clims', [None,None])
        climfactors = kwargs.pop('climfactors', [0,1])
        cmap = kwargs.pop('cmap', None)
        eo = kwargs.pop('eo', 1)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps),eo))
        
        combos = combineArguments(iterproduct, q=q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, climfactors=climfactors, clims=clims, cmap=cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        Qs = list(set([Q[combo[0]] for combo in combos]))
        
        clims = []
        for c in range(len(combos)):
            
            print(f"Computing color limits for plot {c+1} ({Q[combos[c][0]]})...")
            
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
            
            if j%10==0: print("Animating frame {}... ({:.2%})".format(j+1, (j+1)/len(snaps)))
            
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
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Evolution Movie"
            anim.save(self.getPathAndName(filename, ext), writer=writer)
        
        return
    
    
    
    def densityProfileMovie(self, with_slice=True, **kwargs):
        
        """
        Animates the evolution of the soliton density and its radial profile.
        - rmin, rmax, shells: radial range and resolution
        - normalize: animate rho/rho_0 by r/r_c
        """
        
        figsize = kwargs.pop('figsize', ((8,4) if with_slice else (4,4)))
        dpi = kwargs.pop('dpi', 200)
        clims = kwargs.pop('clims', [5,10])
        climfactors = kwargs.pop('climfactors', [0,1])
        axis = kwargs.pop('axis', 1)
        filename = kwargs.pop('filename', "Density Profile Movie")
        save = kwargs.pop('save', True)
        fps = kwargs.pop('fps', 10)
        eo = kwargs.pop('eo', 1)
        snaps = kwargs.pop('snaps', np.arange(0,len(self.snaps),eo))
        
        if clims is None: 
            print("Computing color limits for density animation...")
            clims = getColorLimits(self.get('rho', snaps=snaps), climfactors)
        
        fig, ax = plt.subplots(1, (2 if with_slice else 1), figsize=figsize, dpi=dpi)
        if not hasattr(ax, '__len__'): ax = [ax]
        plt.subplots_adjust(wspace=0.3)
        colorbarmade = False
        
        def animate(i):
            
            if i%10==0: print("Animating frame {}... ({:.2%})".format(i+1, (i+1)/len(snaps)))
        
            nonlocal colorbarmade
            
            s = snaps[i]
            
            ax[-1].clear()
            self.snaps[s].densityProfile(ax=ax[-1], **kwargs)
            
            if with_slice:
                ax[0].clear()
                self.snaps[s].slicePlot('rho', axis, iSlice='max', ax=ax[0], log10=True, colorbar=(not colorbarmade), clims=clims)
                colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(snaps))
        writer = ani.PillowWriter(fps=fps)
        if save: anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
            
        return
    
    
    
    def profileTailOverlays(self, snaps=None, rmin=1, rmax=8, shells=200, **kwargs):
        
        """
        Overlays density profile tails, colored by time.
        - snaps: list of snapshot numbers
        - rmin, rmax, shells: radial domain and resolution
        """
        
        dpi = kwargs.pop('dpi', 200)
        cmap = kwargs.pop('cmap', 'brg')
        eo = kwargs.pop('eo', 1)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', "Profile Tail Overlays")
        if snaps is None: snaps = np.arange(int(self.Nout/4), self.Nout, eo)
        
        fig, ax = plt.subplots(dpi=dpi)
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(self.snaps[snaps[0]].t, self.snaps[snaps[-1]].t)
        for i in range(len(snaps)):
            
            s = self.snaps[snaps[i]]
            
            if i%10==0: print("Plotting tail from Snap {}... ({:.2%})".format(snaps[i], i/len(snaps)))
            
            log10r = np.linspace(np.log10(rmin), np.log10(rmax), shells)
            profile = s.get('profile')
            ax.loglog(10**log10r, 10**profile(log10r), c=cmap(norm(s.t)), **kwargs)
        
        ax.set(xlabel=rf"$r$ {U['length']}", ylabel=rf"$\rho$ {U['density']}", title=r"($f_{15}$" + rf"$ = {self.f15}$)")
        plt.grid(True)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        cbar.set_label("Time " + U['time'])
        
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return
    
    
    
    def smoothTailEvolutionPlot(self, rmin=1, rmax=8, shells=200, neighbors=1, **kwargs):
        
        """
        Plots evolution of power law index of density profile tail, smoothed by SMA of tail density values (not SMA of the index).
        - rmin, rmax, shells: radial domain and resolution
        - neighbors: neighbors to consider in moving average
        """
        
        figsize = kwargs.pop('figsize', (9,3))
        dpi = kwargs.pop('dpi', 200)
        snaprange = kwargs.pop('snaprange', (0,self.Nout))
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', f"{Q['tailindex']} (Smooth - {neighbors}) Evolution Plot")
        
        snaps = np.arange(snaprange[0], snaprange[1])
        t = self.t[snaps][neighbors:-neighbors]
        r = np.logspace(np.log10(rmin), np.log10(rmax), shells)
        rhos = np.zeros((len(snaps), shells))
        
        for i in range(len(snaps)):    
            if i%10==0: print("Retrieving density profile tail from Snap {}... ({:.2%})".format(snaps[i], i/len(snaps)))
            profile = self.snaps[snaps[i]].get('profile')
            rhos[i] = 10**profile(np.log10(r))
            
        
        smooth_rhos = np.zeros((len(snaps)-2*neighbors, shells))
        for i in range(len(smooth_rhos)): smooth_rhos[i] = np.mean(rhos[i:i+2*neighbors+1], axis=0)
        
        def log10PowerLaw(log10r, log10A, n): return log10A + n*log10r
        
        ns = []
        for i in range(len(smooth_rhos)):
            fit = curve_fit(log10PowerLaw, np.log10(r), np.log10(smooth_rhos[i]), p0=[0,-3])
            ns.append(fit[0][1])
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(t, ns, **kwargs)
        ax.set(xlabel=f"Time {U['time']}", ylabel=r"$n_{\mathrm{tail}}$"+f" Smoothened over {neighbors} Point(s)")
        ax.grid()
        
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return t, ns
    
    
    
    def getPathAndName(self, filename, ext):
        
        """
        Retrieves folder path based on saving preferences and creates filename.
        - filename: name to assign file
        - ext: file extension
        """
        
        path = createImageSubfolders(parent=saveToParent, folders=[self.snapdir], mkdir=False)[0]
        if not os.path.isdir(path): path = saveToParent
        path = os.path.join(path, self.dir + " - " + filename + ext)
        
        return path



class Snap():
    
    """
    Extracts and calculates data from one snapshot.
    - snapdir: path to output directory
    - snapnum: assigned number of snapshot
    - lite: do not load psi at initialization
    - loadall: load every supported attribute at initialization
    """
    
    def __init__(self, snapdir, snapnum, lite=False, loadall=False):
        
        self.snapdir = snapdir
        self.num = snapnum
        self.lite = lite
        self.loadall = loadall
        self.filename = "snap{snapnum:04d}.h5".format(snapnum=snapnum)          
        snappath = os.path.join(snapdir, self.filename)
        f = h5py.File(snappath, 'r')
        self.path = snappath
        self.dir = os.path.basename(snapdir)
        
        if not lite: self.psi = np.array(f['psiRe'], dtype=np.float32) + 1j*np.array(f['psiIm'], dtype=np.float32)
        self.t = float(f['time'][0])
        self.m22 = float(f['m22'][0])
        self.m = self.m22 * 8.96215327e-89
        self.f15 = float(self.dir[self.dir.find('f')+1:self.dir.find('L')])
        self.f = self.f15 * 8.05478166e-32
        self.a_s = _a_s(self.f, self.m)                  
        self.Lbox = float(f['Lbox'][0])
        self.N = int(self.dir[self.dir.find('r')+1:])
        self.dx = self.Lbox/self.N
        
        self.crit_M_sol = (_crit_M_sol(self.a_s, self.m) if self.a_s != 0 else np.inf)
        self.crit_rho0 = _crit_rho0(self.f15, self.m22)
        self.crit_r_c = _crit_r_c(self.f15, self.m22)
        self.crit_beta = _crit_beta()
        
        if loadall:
            self.phase = np.angle(self.psi)
            self.rho = np.abs(self.psi)**2
            self.rhobar = np.mean(self.rho)
            self.rhoMax = np.max(self.rho)
            self.iMax = np.array(np.unravel_index(np.argmax(self.rho), self.rho.shape))
            self.profile = self.densityProfile()
            self.fitdict = self.fitProfile()
            self.rho0 = self.fitdict['rho0']
            self.delta = self.fitdict['delta']
            self.r_c = self.fitdict['r_c']
            self.cutoff = self.fitdict['cutoff']
            self.M_sol = self.fitdict['M_sol']
            self.crit_f15 = _crit_f15(self.M_sol, self.m)
            self.beta = self.fitdict['beta']
            self.n = self.fitdict['n']
            self.n_tail = self.fitdict['n_tail']
            self.A_tail = self.fitdict['A_tail']
            self.r_tail = self.fitdict['r_tail']
            self.solitonGOF = self.fitdict['solitonGOF']
            self.tailGOF = self.fitdict['tailGOF']
            self.profileGOF = self.fitdict['profileGOF']
            
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
    
    
    
    def __str__(self): return f"Snapshot {self.num} from output folder {self.dir}."
    
    
     
    def get(self, q, axis=None, project=False, i=None, iSlice=None, log10=False, **kwargs):
        
        """
        Retrieves a quantity (along index kwargs) from the parent snap.
        - q: name of quantity
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        """
        
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
        if q=='t': data = self.t
        if q=='m22': data = self.m22
        if q=='m': data = self.m
        if q=='f15': data = self.f15
        if q=='f': data = self.f
        if q=='a_s': data = self.a_s                         
        if q=='crit_M_sol': data = self.crit_M_sol
        if q=='crit_rho0': data = self.crit_rho0
        if q=='crit_r_c': data = self.crit_r_c
        if q=='crit_beta': data = self.crit_beta
        if q=='Lbox': data = self.Lbox
        if q=='N': data = self.N
        if q=='dx': data = self.dx
        
        if q=='psi':
            try: data = self.psi if full else self.psi[index]
            except:
                f = h5py.File(self.path, 'r')
                psiRe = f['psiRe']
                psiIm = f['psiIm']
                if full:
                    self.psi = np.array(psiRe) + 1j*np.array(psiIm)
                    data = self.psi
                else:
                    data = np.array(psiRe[index]) + 1j*np.array(psiIm[index])
        if q=='phase':
            try: data = self.phase if full else self.phase[index]
            except:
                if full:
                    psi = self.get('psi')
                    self.phase = np.angle(psi)
                    data = self.phase
                else:
                    psi = self.get('psi', i=index)
                    data = np.angle(psi)
        if q=='rho': 
            try: data = self.rho if full else self.rho[index]
            except: 
                if full:
                    psi = self.get('psi')
                    self.rho = np.abs(psi)**2
                    data = self.rho
                else:
                    psi = self.get('psi', i=index)
                    data = np.abs(psi)**2
        if q=='rhobar': 
            try: data = self.rhobar 
            except: 
                rho = self.get('rho')
                self.rhobar = np.mean(rho)
                data = self.rhobar
        if q in ['rhoMax', 'iMax']:
            try: data = self.rhoMax if q=='rhoMax' else self.iMax
            except:
                rho = self.get('rho')
                self.iMax = np.unravel_index(np.argmax(rho), rho.shape)
                self.rhoMax = rho[self.iMax]
                data = self.rhoMax if q=='rhoMax' else self.iMax
        if q=='profile':
            try: data = self.profile
            except:
                self.profile = self.densityProfile(**kwargs)
                data = self.profile
        if q=='fitdict':
            try: data = self.fitdict
            except:
                self.fitdict = self.fitProfile()
                data = self.fitdict
        if q=='rho0':
            try: data = self.rho0
            except:
                fitdict = self.get('fitdict')
                self.rho0 = fitdict['rho0']
                data = self.rho0
        if q=='delta':
            try: data = self.delta
            except:
                fitdict = self.get('fitdict')
                self.delta = fitdict['delta']
                data = self.delta
        if q=='r_c':
            try: data = self.r_c
            except:
                fitdict = self.get('fitdict')
                self.r_c = fitdict['r_c']
                data = self.r_c
        if q=='cutoff':
            try: data = self.cutoff
            except:
                fitdict = self.get('fitdict')
                self.cutoff = fitdict['cutoff']
                data = self.cutoff
        if q=='M_sol':
            try: data = self.M_sol
            except:
                fitdict = self.get('fitdict')
                self.M_sol = fitdict['M_sol']
                data = self.M_sol
        if q=='crit_f15':
            try: data = self.crit_f15
            except:
                M_sol = self.get('M_sol')
                self.crit_f15 = _crit_f15(M_sol, m=self.m)
                data = self.crit_f15
        if q=='beta':
            try: data = self.beta
            except:
                fitdict = self.get('fitdict')
                self.beta = fitdict['beta']
                data = self.beta
        if q=='n':
            try: data = self.n
            except:
                fitdict = self.get('fitdict')
                self.n = fitdict['n']
                data = self.n
        if q=='n_tail':
            try: data = self.n_tail
            except:
                fitdict = self.get('fitdict')
                self.n_tail = fitdict['n_tail']
                data = self.n_tail
        if q=='A_tail':
            try: data = self.A_tail
            except:
                fitdict = self.get('fitdict')
                self.A_tail = fitdict['A_tail']
                data = self.A_tail
        if q=='r_tail':
            try: data = self.r_tail
            except:
                fitdict = self.get('fitdict')
                self.r_tail = fitdict['r_tail']
                data = self.r_tail
        if q=='solitonGOF':
            try: data = self.solitonGOF
            except:
                fitdict = self.get('fitdict')
                self.solitonGOF = fitdict['solitonGOF']
                data = self.solitonGOF
        if q=='tailGOF':
            try: data = self.tailGOF
            except:
                fitdict = self.get('fitdict')
                self.tailGOF = fitdict['tailGOF']
                data = self.tailGOF
        if q=='profileGOF':
            try: data = self.profileGOF
            except:
                fitdict = self.get('fitdict')
                self.profileGOF = fitdict['profileGOF']
                data = self.profileGOF
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
                psi = self.get('psi')
                dpsi = gradient(psi, self.dx)
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
        
        if data is None: raise KeyError(f"Quantity '{q}' is not supported.")
        
        return data
    
    
    
    def singlePlot2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        """
        Plots a single quantity defined throughout the box.
        NOTE: This function is a helper. Use plot2d for same functionality + multi-quantity support.
        - q: name of quantity
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        - ax: axes on which to plot
        """
        
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
        
        data = self.get(q, axis, project, i, iSlice, log10)
        if isinstance(axis, str): axis = "xyz".find(axis)
        if iSlice is None: iSlice = int(self.N/2)-1
        if iSlice == 'max': 
            full_data = self.get(q)
            iSlice = np.unravel_index(np.argmax(full_data), full_data.shape)[axis]
        assert len(np.shape(data))==2, f"Requested data is not 2-dimensional (shape {np.shape(data)})."
        
        data = data[zoom[0]:zoom[1], zoom[2]:zoom[3]]
        extent = kwargs.pop('extent', np.array([zoom[2], zoom[3], zoom[1], zoom[0]])*self.dx)
        
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
    
    
    
    def plot2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        """
        Plots multiple quantities defined throughout the box.
        - q: name of quantity(s)
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        - ax: axes on which to plot
        """
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        clims = kwargs.pop('clims', [None,None])
        zoom = kwargs.pop('zoom', [0,self.N,0,self.N])
        save = kwargs.pop('save', False)
        ext = kwargs.pop('ext', '.pdf')
        filename = kwargs.pop('filename', None)
        wspace = kwargs.pop('wspace', 0.3)
        iterproduct = kwargs.pop('iterproduct', False)
        cmap = kwargs.pop('cmap', None)
        
        combos = combineArguments(iterproduct, q=q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, climfactors=climfactors, clims=clims, cmap=cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        Qs = list(set([Q[combo[0]] for combo in combos]))
        
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
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs}"
            plt.savefig(self.getPathAndName(filename, ext), transparent=True)
        
        return data
    
    
    
    def slicePlot(self, q, axis=1, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        """
        Plots a cross-section of any quantities defined throughout the box.
        - q: name of quantity(s)
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        - ax: axes on which to plot
        """
        
        return self.plot2d(q, axis=axis, i=i, iSlice=iSlice, log10=log10, ax=ax, **kwargs)

        
    
    def projectionPlot(self, q, axis=1, i=None, log10=False, ax=None, **kwargs):
        
        """
        Plots a projection of a quantity defined throughout the box.
        NOTE: By "projection", I mean "maximum along an axis" for now (no weighting).
        - q: name of quantity(s)
        - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        - ax: axes on which to plot
        """
        
        return self.plot2d(q, axis=axis, project=True, i=i, log10=log10, ax=ax, **kwargs)
    
    
    
    def scan3d(self, q, axis=1, i=None, log10=False, **kwargs):
        
        """
        Animates cross-sections of quantities through the box.
        - q: name of quantity
        - index kwargs (axis, i): specify index of multi-dimensional quantity
        - log10: return log (base 10) of quantity
        """
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        clims = kwargs.pop('clims', [None,None])
        cmap = kwargs.pop('cmap', None)
        wspace = kwargs.pop('wspace', 0.3)
        fps = kwargs.pop('fps', 8)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', None)
        iterproduct = kwargs.pop('iterproduct', False)
        eo = kwargs.pop('eo', 1)
        frames = kwargs.pop('frames', np.arange(0, self.N, eo))
        
        combos = combineArguments(iterproduct, q, axis, '/', i, '/', log10, climfactors, clims, cmap)
        combos = [np.array(combo, dtype=object) for combo in combos]
        Qs = list(set([Q[combo[0]] for combo in combos]))
        
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
            
            if j%10==0: print("Animating frame {}... ({:.2%})".format(j+1, j/len(frames)))
            
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
                if len(Qs)==1: Qs = Qs[0]
                filename = f"{Qs} Scan"
            anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
        
        return data
    
    
    
    def densityProfile(self, rmin=None, rmax=None, shells=200, normalize=False, raw=False, recalculate=False, 
                       neighbors=1, rands=1e4, plot=False, fit=False, ax=None, **kwargs):
        
        """
        Computes density profile using sub-pixel density centering and averaging.
        - rmin, rmax, shells: radial domain and resolution
        - normalize: measure rho/rho_0 by r/r_c
        - raw: return raw, discrete data (not interpolation)
        - recalculate: compute profile regardless of past computations
        - neighbors: number of neighbors to consider in sub-pixel centering
        - rands: number of random coordinates used in each shell for sub-pixel averaging
        - plot: show plot along with computation
        - fit: include best fit theoretical soliton profile with plot
        - ax: axes on which to plot
        """
        
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', "Density Profile")
        interpkws = kwargs.pop('interpkws', {'k':5, 's':0.02, 'ext':1})
        legendkws = kwargs.pop('legendkws', {'loc':3, 'fontsize':'x-small'})
        components = kwargs.pop('components', False)
        
        rho, iMax, rho0, r_c = self.get('rho'), self.get('iMax'), None, None
        if normalize or fit: rho0, r_c = self.get('rho0'), self.get('r_c')
        
        loglog_profile = None
        if 'profile' in dir(self): loglog_profile = self.profile
        else: recalculate = True
        
        standard = (rmin is None) and (rmax is None) and (shells==200) and (not raw) and (not normalize) and (neighbors==1) and (rands==1e4)
        mids = rho_r = None
        if recalculate or not standard:
            
            iC = np.array([int(self.N/2)-1]*3)
            x_ = (np.arange(self.N)+0.5)*self.dx
            x, y, z = np.meshgrid(x_, x_, x_)
            xC, yC, zC = x_[iC]
            rho = np.roll(rho, list(iC-iMax), axis=np.arange(3))
            
            iN = slice(iC[0]-neighbors, iC[0]+neighbors+1)
            xN = x_[(iN)]
            mN = np.ravel(rho[iN, iN, iN]*self.dx**3)
            coords = np.array([np.ravel(x) for x in np.meshgrid(xN, xN, xN)])
            xCM, yCM, zCM = np.sum(mN*coords, axis=1)/np.sum(mN)
            
            if rmin is None: rmin = 1e-2
            elif normalize: rmin = rmin*r_c
            if rmax is None: rmax = self.Lbox/2
            elif normalize: rmax = rmax*r_c
            
            mids = np.logspace(np.log10(rmin), np.log10(rmax), shells)
            spacing = np.log10(mids[1])-np.log10(mids[0])
            r = np.logspace(np.log10(mids[0])-spacing/2, np.log10(mids[-1])+spacing/2, shells+1)
            
            rho_r = np.zeros(shells)
            for i in range(shells):
                random_coords = np.random.uniform(-1,1,size=(3, int(rands)))
                norm = np.sqrt(np.sum(random_coords**2, axis=0))
                random_coords = random_coords/norm
                r_for_randoms = np.logspace(np.log10(r[i]), np.log10(r[i+1]), int(rands))
                random_coords = random_coords * r_for_randoms
                random_coords = np.array([random_coords[0]+xCM, random_coords[1]+yCM, random_coords[2]+zCM])
                random_i = np.int32(np.round(random_coords/self.dx-0.5))
                random_i[random_i >= self.N] = self.N-1
                rho_r[i] = np.mean(rho[random_i[0], random_i[1], random_i[2]])
            
            if normalize: rho_r, mids = rho_r/rho0, mids/r_c
            if not raw: 
                loglog_profile = UnivariateSpline(np.log10(mids), np.log10(rho_r), **interpkws)
                if standard: self.profile = loglog_profile
        
        else:
            mids = np.logspace(-2,1,shells)
            rho_r = 10**loglog_profile(np.log10(mids))
            
        if plot or save or ax is not None:
            
            figsize = kwargs.pop('figsize', (6,4))
            dpi = kwargs.pop('dpi', 200)
            lims = kwargs.pop('lims', ([1e-1,1e2,1e-5,2] if normalize else [1e-2,1e1,1e5,2e11]))
            
            if ax is None: fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            label = r'$\rho_{\mathrm{max}}'+' = {:.1e}$ '.format(self.get('rhoMax')) + U['density']
            ax.loglog(mids, rho_r, 'k', label=label, linewidth=5, **kwargs)
            ax.set_xlabel(r"$r$" + normalize*r"$/r_c$" + (not normalize)*f" {U['length']}")
            ax.set_ylabel(r"$\rho(r)$" + normalize*r"$/\rho_0$" + (not normalize)*f" {U['density']}")
            ax.set_title(r"($f_{15}$" + rf"$ = {self.f15}$, $t = {np.round(self.t,3)}$)")
            ax.set(xlim=(lims[0], lims[1]), ylim=(lims[2], lims[3]))
            lims = np.append(ax.get_xlim(), ax.get_ylim())
            
            if fit:
                
                r_fit = np.logspace(np.log10(lims[0]/2), np.log10(lims[1]*2), 100)
                fd = self.get('fitdict')
                sol_fit, tail_fit, rho_fit = None, None, None
                rho_fit = self.theoreticalProfile(r_fit)
                label=r'Profile Fit ($\rho_0 = {:.1e}$, $\delta = {:.3}$, $n = {:.3}$)'.format(fd['rho0'], fd['delta'], fd['n'])
                ax.loglog(r_fit, rho_fit, c='goldenrod',label=label)
                
                if components:
                    sol_fit, tail_fit, rho_fit = self.theoreticalProfile(r_fit, components=True)
                    ax.loglog(r_fit, sol_fit, 'b', label=r'Soliton Component ($r_c = {:.3}$, $M = {:.1e}$, $\beta = {:.3}$)'.format(fd['r_c'], fd['M_sol'], fd['beta']), alpha=0.2)
                    ax.loglog(r_fit, tail_fit, 'r', label=r'Tail Component ($A_t = {:.1e}$, $r_t = {:.3}$, $n_t = {:.3}$)'.format(fd['A_tail'], fd['r_tail'], fd['n_tail']), alpha=0.2)
                
                cutoff = self.get('cutoff')
                plt.vlines((cutoff if normalize else cutoff*r_c), lims[2], lims[3], linestyles='dashed', label=r"Observed Cutoff ({:.4}$r_c$)".format(cutoff), color="rosybrown")
            
            ax.grid()
            plt.legend(**legendkws)
            
            if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return (mids, rho_r) if raw else loglog_profile
    
    
    
    def theoreticalProfile(self, r, rho0=None, delta=None, A_tail=None, n_tail=None, r_tail=None, components=False):
        
        if rho0 is None: rho0 = self.get('rho0')
        if delta is None: delta = self.get('delta')
        if A_tail is None: A_tail = self.get('A_tail')
        if n_tail is None: n_tail = self.get('n_tail')
        if r_tail is None: r_tail = self.get('r_tail')
        b = _beta(rho0, self.f, self.m22)
        r_c = _r_c(rho0, delta, self.m22)
        
        if not hasattr(r, '__len__'): r = np.array([r])
        
        soliton = _rho_sol(r, rho0-A_tail, r_c, b, delta, self.m22)
        tail = _rho_tail(r, r_tail, A_tail, n_tail)
        
        return (soliton, tail, soliton + tail) if components else soliton + tail
        
    
    
    def fitProfile(self, rmin=None, rmax=None, shells=200, plot=False, ax=None, **kwargs):
        
        """
        Fits observed density profile with a soliton curve and a cored power-law tail simultaneously.
        - rmin, rmax, shells: radial domain and resolution
        - plot: show plot with computation
        - ax: axes on which to plot
        """
        
        dpi = kwargs.pop('dpi', 200)
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', "Density Profile Fit")
        p0 = kwargs.pop('p0', [10,1,7,-2,0.1])
        bounds = kwargs.pop('bounds', ([6,0.5,0,-4,1e-2], [15,2,14,4,1]))
        curve_fit_kwargs = kwargs.pop('curve_fit_kwargs', {})
        
        def log10TheoreticalProfile(log10r, log10rho0, delta, log10A, n_tail, r_tail, return_cutoff=False):
            
            rho0 = 10**log10rho0
            b = _beta(rho0, self.f, self.m22)
            r_c = _r_c(rho0, delta, self.m22)
            i1 = np.tanh(b/5)
            i2 = np.tanh(b)
            i3 = np.tanh(np.sqrt(b))**2
            
            soliton = np.log10(10**log10rho0 - 10**log10A) + np.log10((1 + (1+2.60*i1) * 0.091*((10**log10r/r_c)*np.sqrt(1+b))**(2-i2/5))**(-8+22/5*i3))
            #tail = log10A + n*log10r
            tail = log10A + n_tail/2*np.log10((10**log10r/r_tail)**2 + 1)
            
            if return_cutoff: return np.argmin(np.abs(tail - soliton - 1/2))
            
            return np.log10(10**soliton + 10**tail)
        
        if rmin is None: rmin = self.dx*3/2
        if rmax is None: rmax = 4/5*self.Lbox/2
        log10r = np.linspace(np.log10(rmin), np.log10(rmax), shells)
        r = 10**log10r
        profile = self.get('profile')
        log10rho = profile(log10r)
        rho = 10**log10rho
        fit = curve_fit(log10TheoreticalProfile, log10r, log10rho, p0=p0, bounds=bounds, **curve_fit_kwargs)
        
        log10rho0FIT, deltaFIT, log10AFIT, n_tailFIT, r_tailFIT = fit[0]
        rho0FIT, AFIT = np.power(10, [log10rho0FIT, log10AFIT])
        r_cFIT = _r_c(rho0FIT, deltaFIT, self.m22)
        betaFIT = _beta(rho0FIT, self.f, self.m22)
        M_solFIT = _M_sol(rho0FIT, r_cFIT, betaFIT, deltaFIT, self.m22)
        
        rhoFIT = 10**log10TheoreticalProfile(log10r, *fit[0])
        profileGOF = GOF(rho, rhoFIT)
        i_cutoff = log10TheoreticalProfile(log10r, *fit[0], return_cutoff=True)
        cutoff = r[i_cutoff]/r_cFIT
        solitonGOF, tailGOF = np.nan, profileGOF
        if i_cutoff!=0:
            solitonGOF = GOF(rho[:i_cutoff], rhoFIT[:i_cutoff])
            tailGOF = GOF(rho[i_cutoff:], rhoFIT[i_cutoff:])
        n = _n(np.mean([cutoff*r_cFIT, self.Lbox]), r_tailFIT, n_tailFIT)
        
        fitdict = {'rho0':rho0FIT, 'delta':deltaFIT, 'r_c':r_cFIT, 'cutoff':cutoff, 'M_sol':M_solFIT, 'crit_f15':None, 'beta':betaFIT, 'n':n, 'n_tail':n_tailFIT, 'A_tail':AFIT, 'r_tail':r_tailFIT,
                   'solitonGOF':solitonGOF, 'tailGOF':tailGOF, 'profileGOF':profileGOF}    
        
        if plot:
            if ax is None: fig, ax = plt.subplots(dpi=dpi)
            ax.loglog(r, rho, 'k.', label="Measured Density", **kwargs)
            
            rFIT = np.logspace(np.log10(rmin), np.log10(rmax), 100)
            rhoFIT = 10**(log10TheoreticalProfile(np.log10(rFIT), *fit[0]))
            label = r"Fit: $\rho_0 = {:.3e}$, $\delta = {:.3}$, $n = {:.3}$".format(10**log10rho0FIT, deltaFIT, n)
            ax.loglog(rFIT, rhoFIT, 'goldenrod', label=label)
            
            ax.set_xlabel(rf"$r$ {U['length']}")
            ax.set_ylabel(rf"$\rho(r)$ {U['density']}")
            ax.set_title(r"($f_{15}$" + rf"$ = {self.f15}$, $t = {np.round(self.t,3)}$)")
            ax.grid()
            plt.legend()
        
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return fitdict
    
    
    
    def getPathAndName(self, filename, ext):
        
        """
        Retrieves folder path based on saving preferences and creates file name.
        - filename: name to assign file
        - ext: file extension
        """
        
        path = createImageSubfolders(parent=saveToParent, folders=[self.snapdir], mkdir=False)[0]
        if not os.path.isdir(path): path = saveToParent
        append = self.dir + " - " + str(self.num) + " - " + filename + ext
        path = os.path.join(path, append)
        
        return path



"""
HELPER FUNCTIONS

- Functions created for convenience and/or elegance when implemented elsewhere.
"""



def gradient(f, dx, i=[0,1,2]): 
    
    """
    Gets the gradient of a quantity defined in the box.
    - f: 3-dimensional array of data
    - dx: grid spacing
    - i: index(s) of gradient to return
    """
    
    axes = np.arange(3)
    grad = []
    if 0 in i: grad.append((np.roll(f, [0,-1,0], axis=axes) - np.roll(f, [0,1,0], axis=axes))/(2*dx))
    if 1 in i: grad.append((np.roll(f, [-1,0,0], axis=axes) - np.roll(f, [1,0,0], axis=axes))/(2*dx))
    if 2 in i: grad.append((np.roll(f, [0,0,-1], axis=axes) - np.roll(f, [0,0,1], axis=axes))/(2*dx))
    
    return np.array(grad)



def getSlice(data, axis, iSlice='max'):
    
    """
    Given 3-dimensional data, returns a 2-d cross-section along an axis.
    NOTE: set iSlice to 'max' to return slice containing largest value in data
    - data: 3-dimensional array of data
    - axis: axis along which to take cross section
    - iSlice: index of slice to return
    """
    
    if isinstance(axis, str): axis = "xyz".find(axis)
    
    if iSlice=='max': iSlice = np.unravel_index(np.argmax(data), data.shape)[axis]
    if axis==0: data = data[iSlice,:,:]
    if axis==1: data = data[:,iSlice,:]
    if axis==2: data = data[:,:,iSlice]
    
    return data



def getProjection(data, axis):
    
    """
    Given 3-dimensional data, returns a 2-d projection along an axis.
    NOTE: This simply finds the maximum along an axis (no weighting).
    - data: 3-dimensional array of data
    - axis: axis along which to project data
    """
    
    if isinstance(axis, str): axis = "xyz".find(axis)
    
    return np.max(data, axis=axis)



def getColorLimits(data, factors=(0,1)):
    
    """
    Computes color limits of a data set for colored plots.
    - data: N-dimensional numerical data set
    - factors: 2-tuple of floats indicating data values of lowest and highest color, relative to the min and max of data.
    """
    
    low, high = np.min(data), np.max(data)
    clims = (low + factors[0]*(high-low), high - (1-factors[1])*(high-low))
    
    return clims



def combineArguments(iterproduct=False, sim='/', snap='/', q='/', axis='/', project='/', i='/', iSlice='/', log10='/', climfactors='/', clims='/', cmap='/', c='/', smooth='/'):
    
    """
    Given ragged assortment of arguments, returns coherent lists of arguments for execution in multi-quantity plotting functions.
    - iterproduct: return all combinations of arguments (if False, do not return "cross" combinations)
    - other kwargs: arguments to combine, either in their natural type or lists
    """
    
    if not hasattr(sim, '__len__'): sim = [sim]
    if not hasattr(snap, '__len__'): snap = [snap]
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
    if isinstance(smooth, int) or smooth is None: smooth = [smooth]
    
    l = {'sim':sim, 'snap':snap, 'q':q, 'axis':axis, 'project':project, 'i':i, 'iSlice':iSlice, 'log10':log10, 'climfactors':climfactors, 'clims':clims, 'cmap':cmap, 'c':c, 'smooth':smooth}
    args = [l[j] for j in l if l[j]!='/']
    if len(set([len(arg) for arg in args]))>2: iterproduct = True
    
    combos = []
    if iterproduct:
        combos = list(itertools.product(*args))
    else:
        maxLen = np.max([len(arg) for arg in args])
        for i in range(maxLen):
            combo = [(arg[i] if len(arg)==maxLen else arg[0]) for arg in args]
            combos.append(combo)
    
    return combos



def unitsFactor(u, to='SI'):
    
    """
    Catalog of conversion factors from code units to other units.
    - u: physical concept (e.g.: 'time')
    - to: unit to which to convert (from code units, e.g.: 'Gyr')
    """
    
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
    
    if factor==1: raise KeyError("Requested unit is not yet supported.")
    
    return factor



def SMA(data, n):
    
    """
    Simple moving average of 1-d list of data.
    - data: list of numbers
    - n: neighbors over which to smoothen
    """
    
    smoothed = data
    if n>0:
        smoothed[:n] = np.repeat(np.nan, n)
        smoothed[-n:] = np.repeat(np.nan, n)
        smoothed[n:-n] = [np.mean(data[i-n:i+n+1]) for i in range(n,len(data)-n)]
    
    return smoothed



def GOF(data, fit):
    
    """
    Goodness of fit metric between discrete arrays of data and fitted values.
    - data: list of values
    - fit: list of fit values (same length as data)
    """
    
    return 1/len(data) * np.sum(np.array(np.log(data) - np.log(fit))**2)



"""
FORMULAE

- Deriving quantities from known formulae.
"""



def _a_s(f, m=8.96215327e-89):
    return hbar*c**3*m/(32*pi*f**2)

def _f(a_s, m=8.96215327e-89, normalize=False):
    return np.sqrt(hbar*c**3*m/(32*pi*a_s))/([1,8.05478166e-32][normalize])

def _rho0(r_c, delta=1, m22=1):
    return 1.9e7*delta*r_c**(-4)*m22**(-2)

def _r_c(rho0, delta=1, m22=1):
    return (rho0/(1.9e7*delta))**(-1/4)*(m22)**(-1/2)

def _M_sol(rho0, r_c=None, beta=0, delta=1, m22=1):
    if beta==0: 
        if r_c is None: r_c = _r_c(rho0, 1, m22)
        return 11.6*delta*rho0*r_c**3
    else:
        dMdr = lambda r, rho0, r_c, beta, delta, m22: _rho_sol(r, rho0, r_c, beta, delta, m22)*4*pi*r**2
        return quad(dMdr, 0, np.inf, args=(rho0, r_c, beta, delta, m22))[0]

def _beta(rho0, f, m22=1):
    return 1.6e-12/m22 * rho0**(1/2) * hbar*c**5/(32*pi*G*f**2)

def _rho_sol(r, rho0, r_c=None, beta=0, delta=1, m22=1):
    
    if r_c is None: r_c = _r_c(rho0, delta, m22)
    
    i1 = np.tanh(beta/5)
    i2 = np.tanh(beta)
    i3 = np.tanh(np.sqrt(beta))**2
    
    return rho0*(1 + (1+2.60*i1) * 0.091*((r/r_c)*np.sqrt(1+beta))**(2-i2/5))**(-8+22/5*i3)

def _rho_tail(r, r_tail, A, n):
    return A*((r/r_tail)**2 + 1)**(n/2)

def _n(r, r_tail, n_tail):
    return n_tail*(r/r_tail)**2 / (1 + (r/r_tail)**2)

def _crit_M_sol(a_s, m=8.96215327e-89):
    return 1.012*hbar/np.sqrt(G*m*a_s)

def _crit_rho0(f15, m22=1):
    return 1.2e9*m22**2*f15**4

def _crit_r_c(f15, m22=1):
    return 0.18/(m22*f15)

def _crit_beta():
    return 0.3

def _crit_a_s(M_sol, m=8.96215327e-89):
    return 1/(G*m)*(M_sol/(1.012*hbar))**(-2)

def _crit_f15(M_sol, m=8.96215327e-89):
    return _f(_crit_a_s(M_sol, m), normalize=True)

def _v_c(r_c, m=8.96215327e-89):
    return 2*pi/7.5*hbar/(m*r_c)

def _nu(rho0):
    return 10.94*(rho0/1e9)**(1/2)



"""
GENERALIZED PLOTTING & ANIMATION

- Compare images and evolving quantities between different simulations.
- No functionality lost from Sim and Snap attributed functions.
"""



def plot2d(snap, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs): 
    
    """
    Plots multiple quantities defined throughout the box.
    - q: name of quantity(s)
    - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
    - log10: return log (base 10) of quantity
    - ax: axes on which to plot
    """
    
    dpi = kwargs.pop('dpi', 200)
    climfactors = kwargs.pop('climfactors', [0,1])
    clims = kwargs.pop('clims', [None,None])
    zoom = kwargs.pop('zoom', None)
    save = kwargs.pop('save', False)
    ext = kwargs.pop('ext', '.pdf')
    filename = kwargs.pop('filename', None)
    wspace = kwargs.pop('wspace', 0.3)
    iterproduct = kwargs.pop('iterproduct', False)
    cmap = kwargs.pop('cmap', None)
    
    combos = combineArguments(iterproduct, snap=snap, q=q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, climfactors=climfactors, clims=clims, cmap=cmap)
    combos = [np.array(combo, dtype=object) for combo in combos]
    Qs = list(set([Q[combo[1]] for combo in combos]))
    
    if ax is None:
        fig, ax = plt.subplots(1, len(combos), figsize=kwargs.pop('figsize', (4*len(combos), 4)), dpi=dpi)
        fig.subplots_adjust(wspace=wspace)
    if not hasattr(ax, '__len__'): ax = np.array([ax])
    
    data = []
    for j in range(len(combos)):
        snap, q, axis, project, i, iSlice, log10, climfactors, clims, cmap = combos[j]
        if zoom is None: zoom = [0,snap.N,0,snap.N]
        data.append(snap.singlePlot2d(q, axis, project, i, iSlice, log10, climfactors=climfactors, clims=clims, cmap=cmap, ax=ax[j], zoom=zoom, **kwargs))
    
    if save: 
        if filename is None:
            if len(Qs)==1: Qs = Qs[0]
            filename = f"{Qs}"
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        plt.savefig(path, transparent=True)
    
    return data



def evolutionPlot(sim, q, i=None, log10=False, ax=None, **kwargs):
    
    """
    Plots the evolution of any scalar-valued quantities.
    - sim: Sim object(s)
    - q: name of quantity(s)
    - i: specifies index of multi-dimensional quantity
    - log10: return log (base 10) of quantity
    - ax: axes on which to plot
    """
    
    figsize = kwargs.pop('figsize', (9,3))
    dpi = kwargs.pop('dpi', 200)
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', None)
    ext = kwargs.pop('ext', '.pdf')
    c = kwargs.pop('c', None)
    iterproduct = kwargs.pop('iterproduct', False)
    legendkws = kwargs.pop('legendkws', {'fontsize':'small'})
    
    combos = combineArguments(iterproduct, sim=sim, q=q, i=i, log10=log10, c=c)
    Qs = list(set([Q[combo[1]] for combo in combos]))
    
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
    
    t, data = [], []
    for j in range(len(combos)):
        
        sim, q, i, log10, c = combos[j]
        
        print(f"Plotting quantity {j+1}: {Q[q]}...")
        
        t_temp, data_temp = sim.evolutionPlot(q, i=i, log10=log10, c=c, ax=ax, save=False, annotate=False, **kwargs)
        t.append(t_temp), data.append(data_temp)
    
    legend = plt.legend(**legendkws)
    for c in range(len(combos)):
        text = legend.get_texts()[c].get_text() + f" ({combos[c][0].dir})"
        legend.get_texts()[c].set_text(text)
    ylabel = kwargs.pop('ylabel', ("Multiple Quantities" if len(Qs)>1 else Qs[0] + (i is not None)*f" {i}" + f" {U.get(U.get(q), '')}"))
    ax.set(xlabel=f"Time {U['time']}", ylabel=ylabel, title=None)
    plt.grid(True)
    
    if save: 
        if filename is None:
            if len(Qs)==1: Qs = Qs[0]
            filename = f"{Qs} Evolution Plot"
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        plt.savefig(path, transparent=True)
    
    return t, data



def evolutionMovie(sim, q, axis=1, project=False, i=None, iSlice=None, log10=False, **kwargs):
    
    """
    Animates the evolution of any quantities defined throughout the box.
    - snapdir: full path to output directory(s)
    - q: name of quantity(s)
    - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
    - log10: return log (base 10) of quantity
    """
    
    dpi = kwargs.pop('dpi', 200)
    wspace = kwargs.pop('wspace', 0.3)
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', None)
    ext = kwargs.pop('ext', '.gif')
    fps = kwargs.pop('fps', 20)
    iterproduct = kwargs.pop('iterproduct', False)
    clims = kwargs.pop('clims', [None,None])
    climfactors = kwargs.pop('climfactors', [0,1])
    cmap = kwargs.pop('cmap', None)
    eo = kwargs.pop('eo', 1)
    snaps = kwargs.pop('snaps', None)
    
    combos = combineArguments(iterproduct, sim=sim, q=q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, climfactors=climfactors, clims=clims, cmap=cmap)
    combos = [np.array(combo, dtype=object) for combo in combos]
    Qs = list(set([Q[combo[1]] for combo in combos]))
    snaps = np.arange(0, combos[0][0].Nout, eo) if snaps is None else snaps[::eo]
    
    clims = []
    for c in range(len(combos)):
        
        print(f"Computing color limits for plot {c+1} ({combos[c][0].dir}, {Q[combos[c][1]]})...")
        
        if combos[c][-2]==[None,None] or combos[c][-2]==[]:
            _sim, _q, _axis, _project, _i, _iSlice, _log10, _climfactors, _1, _2 = combos[c]
            data = _sim.get(_q, snaps=snaps, axis=_axis, project=_project, i=_i, iSlice=_iSlice, log10=_log10)
            clims.append(getColorLimits(data, _climfactors))
        else:
            clims.append(combos[c][-2])
    
    fig, ax = plt.subplots(1, len(combos), figsize=kwargs.pop('figsize', (4*len(combos),4)), dpi=dpi)
    if not hasattr(ax, '__len__'): ax = [ax]
    fig.subplots_adjust(wspace=wspace)
    colorbarmade = False
    
    def animate(j):
        
        if j%10==0: print("Animating frame {}... ({:.2%})".format(j+1, (j+1)/len(snaps)))
        
        nonlocal colorbarmade
        
        [ax[k].clear() for k in range(len(ax))]
        
        s = [combo[0].snaps[snaps[j]] for combo in combos]
        plot2d(s, q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, ax=ax, colorbar=(not colorbarmade), clims=clims, **kwargs)
        colorbarmade = True
        
        return
    
    anim = ani.FuncAnimation(fig, animate, frames=len(snaps))
    writer = ani.PillowWriter(fps=fps)
    
    if save: 
        if filename is None:
            if len(Qs)==1: Qs = Qs[0]
            filename = f"{Qs} Evolution Movie"
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        anim.save(path, writer=writer)
    
    return



def diff2d(snap1, snap2, q, ratio=False, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
    
    """
    Plots the difference or ratio of a quantity defined throughout the box between two snapshots.
    - snap1, snap2: Snap objects
    - q: name of quantity
    - ratio: plot the difference normalized by q in snap2
    - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
    - log10: return log (base 10) of quantity
    - ax: axes on which to plot
    """
    
    assert (snap1.N == snap2.N) and (snap1.dx == snap2.dx), "Snapshots must have equal resolution."
    assert iSlice != 'max', "Max-valued slices are not equal, in general."
    
    dpi = kwargs.pop('dpi', 200)
    zoom = kwargs.pop('zoom', [0,snap1.N,0,snap1.N])
    save = kwargs.pop('save', False)
    ext = kwargs.pop('ext', '.pdf')
    filename = kwargs.pop('filename', None)
    climfactors = kwargs.pop('climfactors', [0,1])
    clims = kwargs.pop('clims', None)
    cmap = kwargs.pop('cmap', 'bone')
    colorbar = kwargs.pop('colorbar', True)
    title = kwargs.pop('title', None)
    colorscale = kwargs.pop('colorscale', 'linear')
    absolute = kwargs.pop('absolute', False)
    
    if log10 and ratio and colorscale=='linear':
        if (input("Did you want the colorbar on a log scale? ('y' or 'n')\n") == 'y'): 
            colorscale, log10, absolute = 'log', False, True
        else:
            raise Exception("Ratio of logs is not physically meaningful.")
    
    if ax is None: fig, ax = plt.subplots(dpi=dpi)
    
    q1 = snap1.get(q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10)
    q2 = snap2.get(q, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10)
    data = q1 - q2
    if ratio: data = data/q2
    if absolute or colorscale=='log': data = np.abs(data)
    
    if isinstance(axis, str): axis = "xyz".find(axis)
    if iSlice is None: iSlice = int(snap1.N/2)-1
    assert len(np.shape(data))==2, f"Requested data is not 2-dimensional (shape {np.shape(data)})."
    
    data = data[zoom[0]:zoom[1], zoom[2]:zoom[3]]
    extent = kwargs.pop('extent', np.array([zoom[2], zoom[3], zoom[1], zoom[0]])*snap1.dx)
    if clims is None: clims = getColorLimits(data, climfactors)
    norm = mpl.colors.LogNorm(vmin=clims[0], vmax=clims[1]) if colorscale=='log' else None
    
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, **kwargs)
    axes = ['yz', 'zx', 'xy'][axis]
    ax.set(xlabel=rf"${axes[0]}$ [kpc]", ylabel=rf"${axes[1]}$ [kpc]")
    if title is None: 
        title_f15 = r"($f_{15}$" + rf"$ = ({snap1.f15}, {snap2.f15})$, "
        title_slice = ("" if project else rf"${'xyz'[axis]} = {np.round((iSlice+0.5)*snap1.dx,2)}$, ")
        title_t = rf"$t = ({np.round(snap1.t,3)}, {np.round(snap2.t,3)})$)"
        title = title_f15 + title_slice + title_t
    ax.set_title(title)
    
    if colorbar is True:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(("Ratio" if ratio else "Difference") + " of " + log10*r'$\log_{10}($' + Q[q] + log10*')' + f" {U.get(U.get(q), '')}")
    
    suptitle = (absolute or colorscale=='log')*"Absolute " + ("Ratio" if ratio else "Difference") + " of" + log10*" Log" + f" {Q[q]} Values"
    plt.suptitle(suptitle)
    
    if save:
        if filename is None: filename = suptitle
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        plt.savefig(path, transparent=True)
    
    return data



def differencePlot(sim1, sim2, q, ratio=False, i=None, log10=False, ax=None, **kwargs):
    
    """
    Plots the evolution of the difference or ratio of a quantity extracted from two different simulations.
    - sim1, sim2: Sim objects
    - q: name of quantity
    - ratio: plot the difference normalized by q in sim2
    - i: specifies index of multidimensional quantity
    - log10: return log (base 10) of quantity
    - ax: axes on which to plot
    """
    
    assert not isinstance(q, list), "Plot one quantity at a time."
    
    figsize = kwargs.pop('figsize', (9,3))
    dpi = kwargs.pop('dpi', 200)
    yscale = kwargs.pop('yscale', 'linear')
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', None)
    ext = kwargs.pop('ext', '.pdf')
    eo = kwargs.pop('eo', 1)
    snaps = kwargs.pop('snaps', None)
    smooth = kwargs.pop('smooth', None)
    
    t1, t2 = (sim1.t, sim2.t) if snaps is None else (sim1.t[snaps], sim2.t[snaps])
    assert np.array_equal(np.round(t1,3), np.round(t2,3)), "Different time arrays."
    t = t1[::eo]
    if snaps is None: snaps = np.arange(0, sim1.Nout, eo)
    q1 = sim1.get(q, snaps=snaps, i=i, log10=log10)
    q2 = sim2.get(q, snaps=snaps, i=i, log10=log10)
    
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
    
    data = q1 - q2
    if ratio: data = data/q2
    if isinstance(smooth, int): data = SMA(data, smooth)
    if yscale == 'log': data = np.abs(data)
    ax.plot(t, data, **kwargs)
    ylabel = ratio*"Normalized " + f"Difference: {Q[q]}"
    title = r"($f_{15} = $" + rf"${sim1.f15}$, ${sim2.f15}$)"
    ax.set(xlabel=f"Time {U['time']}", ylabel=ylabel, title=title, yscale=yscale)
    plt.grid(True)
    
    if save: 
        if filename is None: filename = Q[q] + ratio*" Normalized" + " Difference Plot"
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        plt.savefig(path, transparent=True)
    
    return t, data



def differenceMovie(sim1, sim2, q, ratio=False, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
    
    """
    Animates the evolution of the difference (or ratio) between the same quantity from different simulations.
    - sim1, sim2: Sim objects
    - q: name of quantity
    - ratio: plot the difference normalized by q in sim2
    - index kwargs (axis, project, i, iSlice): specify index of multi-dimensional quantity
    - log10: return log (base 10) of quantity
    - ax: axes on which to plot
    """
    
    dpi = kwargs.pop('dpi', 200)
    save = kwargs.pop('save', True)
    filename = kwargs.pop('filename', None)
    ext = kwargs.pop('ext', '.gif')
    fps = kwargs.pop('fps', 20)
    eo = kwargs.pop('eo', 1)
    
    assert (sim1.N == sim2.N) and (sim1.dx == sim2.dx), "Simulations must have equal resolution."
    assert np.array_equal(np.round(sim1.t,3), np.round(sim2.t,3)), "Simulation snapshots must have identical timestamps."
    
    snaps = kwargs.pop('snaps', np.arange(0,sim1.Nout,eo))
    
    fig, ax = plt.subplots(dpi=dpi)
    colorbarmade = False
    
    def animate(j):
        
        if j%10==0: print("Animating frame {}... ({:.2%})".format(j+1, (j+1)/len(snaps)))
        
        nonlocal colorbarmade
        
        ax.clear()
        
        snap1, snap2 = sim1.snaps[snaps[j]], sim2.snaps[snaps[j]]
        diff2d(snap1, snap2, q, ratio=ratio, axis=axis, project=project, i=i, iSlice=iSlice, log10=log10, ax=ax, colorbar=(not colorbarmade), **kwargs)
        colorbarmade = True
        
        return
    
    anim = ani.FuncAnimation(fig, animate, frames=len(snaps))
    writer = ani.PillowWriter(fps=fps)
        
    if save: 
        if filename is None: filename = f"{Q[q]} Difference Movie"
        dirname = createImageSubfolders(parent=saveToParent, mkdir=False, folders=[], mkcomparisons=True)[0]
        path = os.path.join(dirname, filename + ext)
        anim.save(path, writer=writer)
    
    return




















"""
BACKGROUND PROCESSES

- Handles subfolder creation.
"""



def createImageSubfolders(parent=os.getcwd(), folders=folders, mkdir=True, mkcomparisons=True):
    
    """
    Creates subfolders to organize saved images within given parent directory.
    - parent: directory in which to create subfolders
    - folders: output directories to associate subfolders
    - mkdir: make the directory (if False, only returns folder name)
    - mkcomparisons: make comparisons subfolder
    """
    
    subfolders = []
    for folder in folders:
        basename = os.path.basename(folder)
        dirname = None
        if onlyNameF15:
            f15 = float(basename[basename.find('f')+1:basename.find('L')])
            dirname = os.path.join(parent, "sifdm - f15 - {:.3}".format(f15))
        else:
            dirname = os.path.join(parent, "sifdm - "+os.path.basename(folder))
        subfolders.append(dirname)
        if not os.path.isdir(dirname) and mkdir: 
            os.mkdir(dirname)
    if mkcomparisons:
        dirname = os.path.join(parent, "sifdm - Comparisons")
        subfolders.append(dirname)
        if mkdir and not os.path.isdir(dirname): os.mkdir(dirname)
            
    return subfolders



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
    
        
        
        
        