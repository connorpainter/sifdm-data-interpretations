## SIFDM Data Interpretations:
##
## Created by Connor Painter on 9/27/21 to do elementary analysis on the 
## outputs of SIFDM MATLAB code by Philip Mocz with Python.
## Last updated: 10/15/21



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import h5py
import os
import time
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


## f15 = ?; L = ?; T = ?; Nout = ?; N (res) = ?
d1 = r"/paste/here" ## Paste a full path to a simulation output folder!

## f15 = ?; L = ?; T = ?; Nout = ?; N (res) = ?
d2 = r"/paste/here"

## f15 = ?; L = ?; T = ?; Nout = ?; N (res) = ?
##d3 = r""

## f15 = ?; L = ?; T = ?; Nout = ?; N (res) = ?
##d4 = r""



folders = [d1, d2]



## Preferences for saved images



createImgSubfolders = True
saveToParent = os.getcwd()



## Useful arrays and dictionaries



Q = {'':'', 'snapdir':'Full Path of Snap-Enclosing Directory', 'num':'Snap Number', 'filename':'Snap File Name', 'path':'Full Path of Snap', 'dir':'Snap-Enclosing Directory', 
     'psi':'Wavefunction', 't':'Current Time', 'm22':'Mass [10^-22 eV]', 'm':'Mass', 'f15':'Self-Interaction Constant [10^15 GeV]', 'f':'Self-Interaction Constant', 'Lbox':'Simulation Box Length', 'N':'Resolution', 'dx':'Grid Spacing',
     'phase':'Wavefunction Phase', 'rho':'Density', 'rhobar':'Mean Density', 'rho0':'Maximum Density', 'i0':'Maximum Density Index', 'rc':'Soliton Core Radius', 'V':'Potential', 'v':'Madelung Velocity', 'v2':'Madelung Velocity Magnitude Squared', 'M':'Mass', 'W':'Potential Energy', 
     'Kv':'Energy from Movement', 'Krho':'Energy from Density', 'KQ':'Energy from Quantum Potential', 'L':'Angular Momentum', 'L2':'Angular Momentum Magnitude Squared'}
Q0 = ['t', 'm22', 'm', 'f15', 'f', 'Lbox', 'N', 'dx', 'rhobar', 'rho0', 'rc', 'M', 'W', 'Kv', 'Krho', 'KQ', 'L2']
Q1 = ['L', 'i0']
Q2 = []
Q3 = ['psi', 'phase', 'rho', 'V', 'v2']
Q4 = ['v']
U = {'length':'[kpc]', 'mass':r"[$M_{\odot}$]", 'energy':r"[$M_{\odot}\mathrm{(km/s)}^2$]", 'time':"[kpc/(km/s)]", 'density':r"[$M_{\odot}/\mathrm{kpc}^3$]", 'velocity':"[km/s]",
     't':'time', 'm':'mass', 'f':'energy', 'Lbox':'length', 'dx':'length', 'rho':'density', 'rhobar':'density', 'rho0':'density', 'rc':'length', 'Kv':'energy', 'Krho':'energy', 'KQ':'energy'}
C = {'phase':'bwr', 'rho':'inferno', 'V':'cividis', 'v':'hot', 'v2':'hot'}



## Constants



hbar = 1.71818131e-87                                                           # hbar / (mass of sun * (km/s) * kpc)
G = 4.3022682e-6                                                                # G/((km/s)^2*kpc/mass of sun)
c = 299792.458                                                                  # c / (km/s)
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
        self.t = self.get('t')
        self.m22 = self.snaps[0].m22
        self.m = self.snaps[0].m
        self.f15 = self.snaps[0].f15
        self.f = self.snaps[0].f
        self.Lbox = self.snaps[0].Lbox
        self.N = self.snaps[0].N
        self.dx = self.snaps[0].dx
        self.Nout = len(self.snaps)
        
        return
    
    
    
    ## Retrieves a quantity from every snap (faster the 2nd time).
    def get(self, q, i=None, log10=False):
        
        return [self.snaps[j].get(q, i, log10) for j in range(len(self.snaps))]
    
    
    
    ## Plots evolution of a given quantity that is scalar-valued at every snap.
    def evolutionPlot(self, q, i=None, log10=False, ax=None, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', Q[q] + " Evolution Plot")
        
        if ax is None:
            plt.figure(dpi=dpi)
            ax = plt.gca()
        
        data = self.get(q, i, log10)
        ax.plot(self.t, data, **kwargs)
        ax.set(xlabel=f"Time {U['time']}", ylabel=Q[q] + f" {U.get(U.get(q), '')}", title=r"($f_{15} = $" + rf"${self.f15}$)")
        ax.grid()
        
        if save is True: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return data
    
    
    
    ## Animates the evolution of a given quantity defined throughout the box.
    def evolutionMovie2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        filename = kwargs.pop('filename', Q[q] + " Evolution Movie")
        save = kwargs.pop('save', True)
        fps = kwargs.pop('fps', 20)
        
        data = self.get(q, i)
        data = np.array([ (np.sum(data[t], axis=axis) if project else getSlice(data[t], axis, iSlice)) for t in range(len(data)) ])
        if log10: data = np.log10(data)
        
        fig, ax = plt.subplots(dpi=dpi)
        clims = kwargs.pop('clims', getColorLimits(data, factors=climfactors))
        colorbarmade = False
        
        def animate(j):
            
            if j%10==0: print(f"Animating frame {j+1}/{len(self.t)}")
            
            nonlocal colorbarmade
            
            ax.clear()
            
            self.snaps[j].plot2d(q, axis, project, i, iSlice, log10, ax, colorbar=(not colorbarmade), clims=clims, **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(self.t))
        writer = ani.PillowWriter(fps=fps)
        if save: anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
        
        return
    
    
    
    ## Animates the evolution of the soliton density profile.
    def densityProfileMovie(self, shells=20, normalize=True, fit=False, **kwargs):
        
        figsize = kwargs.pop('figsize', (8,4))
        dpi = kwargs.pop('dpi', 200)
        clims = kwargs.pop('clims', [5,10])
        filename = kwargs.pop('filename', "Density Profile Movie")
        save = kwargs.pop('save', True)
        fps = kwargs.pop('fps', 10)
        
        rho_full = self.get('rho')
        
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(wspace=0.3)
        colorbarmade = False
        
        def animate(i):
            
            if i%10==0: print(f"Animating frame {i+1}/{len(rho_full)}")
        
            nonlocal colorbarmade
            
            ax[0].clear()
            ax[1].clear()
            
            self.snaps[i].slicePlot('rho', 'y', ax=ax[0], log10=True, colorbar=(not colorbarmade), clims=clims, **kwargs)
            self.snaps[i].densityProfile(shells, normalize, fit=fit, plot=False, ax=ax[1], **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=len(rho_full))
        writer = ani.PillowWriter(fps=fps)
        if save: anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
            
        return
    
    
    
    def getPathAndName(self, filename, ext):
        
        path = os.path.join(saveToParent, self.dir + " Images")
        if not os.path.isdir(path): path = saveToParent
        path = os.path.join(path, self.dir + " - " + filename + ext)
        
        return path



## Conveniently extracts and calculates data from one snapshot.
## Analogous to Philip's READSNAP.m helper function.
## Toggle loadall to choose how much data to extract from snapshot initially.
## get(): existence of attribute is checked first, then calculated and stored 
##        if necessary.
class Snap():
    
    def __init__(self, snapdir, snapnum, loadall=False):
        
        self.snapdir = snapdir
        self.num = snapnum
        self.filename = "snap{snapnum:04d}.h5".format(snapnum=snapnum)          ## .format to get around numba
        snappath = os.path.join(snapdir, self.filename)
        f = h5py.File(snappath, 'r')
        self.path = snappath
        self.dir = os.path.basename(snapdir)
        
        self.psi = np.array(f['psiRe']) + 1j*np.array(f['psiIm'])
        self.t = float(f['time'][0])
        self.m22 = float(f['m22'][0])
        self.m = self.m22 * 8.96215327e-89
        self.f15 = float(self.dir[self.dir.find('f')+1:self.dir.find('L')])
        self.f = self.f15 * 8.05478166e-32                                      # 10^15 GeV/((km/s)^2*mass of sun)
        self.Lbox = float(f['Lbox'][0])
        self.N = np.shape(self.psi)[0]
        self.dx = self.Lbox/self.N
        
        self.all_loaded = loadall
        if loadall:
            self.phase = np.angle(self.psi)
            self.rho = np.abs(self.psi)**2
            self.rhobar = np.mean(self.rho)
            self.rho0 = np.max(self.rho)
            self.i0 = np.array(np.unravel_index(np.argmax(self.rho), self.rho.shape))
            self.rc = (self.rho0/3.1e6)**(-1/4)*(2.5/self.m22)**(1/2)
            
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
            v[v<= pi] = v[v<= pi] + 2*pi
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
            
            ## Angular momentum.
            x = (np.arange(self.N) + 1/2)*self.dx - self.Lbox/2
            x, y, z = np.meshgrid(x, x, x)
            Lx = np.sum(self.rho * (self.v[2]*y - self.v[1]*z)) * self.dx**3
            Ly = np.sum(self.rho * (self.v[0]*z - self.v[2]*x)) * self.dx**3
            Lz = np.sum(self.rho * (self.v[1]*x - self.v[0]*y)) * self.dx**3
            self.L = np.array([Lx, Ly, Lz])
            self.L2 = np.sum(self.L**2, axis=0)
        
        return
    
    
    
    ## Retrieves and saves a quantity attributed to the snap (faster the 2nd time).
    def get(self, q, i=None, log10=False):
        
        data = None
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
        if q=='Lbox': data = self.Lbox
        if q=='N': data = self.N
        if q=='dx': data = self.dx
        
        if q=='phase': 
            try: data = self.phase
            except: 
                self.phase = np.angle(self.psi)
                data = self.phase
        if q=='rho': 
            try: data = self.rho
            except: 
                self.rho = np.abs(self.psi)**2
                data = self.rho
        if q=='rhobar': 
            try: data = self.rhobar 
            except: 
                self.rhobar = np.mean(np.abs(self.psi)**2)
                data = self.rhobar
        if q=='rho0':
            try: data = self.rho0
            except:
                rho = self.get('rho')
                self.rho0 = np.max(rho)
                data = self.rho0
        if q=='i0':
            try: data = self.i0
            except:
                rho = self.get('rho')
                self.i0 = np.array(np.unravel_index(np.argmax(rho), rho.shape))
                data = self.i0
        if q=='rc':
            try: data = self.rc
            except:
                rho0 = self.get('rho0')
                self.rc = (rho0/3.1e6)**(-1/4)*(2.5/self.m22)**(1/2)
                data = self.rc
        if q=='V':
            try: data = self.V
            except:
                rho = self.get('rho')
                k = (2*pi/self.Lbox) * np.arange(-self.N/2, self.N/2)
                kx, ky, kz = np.meshgrid(k, k, k)
                kSq = np.fft.fftshift(kx**2 + ky**2 + kz**2)
                Vhat = -np.fft.fftn(4*pi*G*(rho - np.mean(rho)))/(kSq + (kSq==0))
                V = np.fft.ifftn(Vhat)
                self.V = np.real(V - np.mean(V))
                data = self.V
        if q=='v':
            try: data = self.v
            except:
                phase = self.get('phase')
                v = gradient(phase, self.dx)*(2*self.dx)
                v[v > pi] = v[v > pi] - 2*pi
                v[v<= pi] = v[v<= pi] + 2*pi
                self.v = v/(2*self.dx)*(hbar/self.m)
                data = self.v
        if q=='v2':
            try: data = self.v2
            except:
                v = self.get('v')
                self.v2 = np.sum(v**2, axis=0)
                data = self.v2
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
                v = self.get('v')
                self.Kv = np.sum(rho*(v[0]**2 + v[1]**2 + v[2]**2)/2) * self.dx**3
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
        if q=='L':
            try: data = self.L
            except:
                rho = self.get('rho')
                v = self.get('v')
                x = (np.arange(self.N) + 1/2)*self.dx - self.Lbox/2
                x, y, z = np.meshgrid(x, x, x)
                Lx = np.sum(self.rho * (self.v[2]*y - self.v[1]*z)) * self.dx**3
                Ly = np.sum(self.rho * (self.v[0]*z - self.v[2]*x)) * self.dx**3
                Lz = np.sum(self.rho * (self.v[1]*x - self.v[0]*y)) * self.dx**3
                self.L = np.array([Lx, Ly, Lz])
                data = self.L
        if q=='L2':
            try: data = self.L2
            except:
                L = self.get('L')
                self.L2 = np.sum(L**2)
                data = self.L2
        
        if i is not None: data = data[i]
        if log10: data = np.log10(data)
        
        return data
    
    
    
    ## Plots a given quantity defined throughout the box.
    ## Generalizes slicePlot and projectionPlot; use any of them.
    def plot2d(self, q, axis=1, project=False, i=None, iSlice=None, log10=False, ax=None, **kwargs):
        
        figsize = kwargs.pop('figsize', None)
        dpi = kwargs.pop('dpi', 200)
        zoom = kwargs.pop('zoom', [0,self.N,0,self.N])
        cmap = kwargs.pop('cmap', C.get(q, 'viridis'))
        climfactors = kwargs.pop('climfactors', [0,1])
        colorbar = kwargs.pop('colorbar', True)
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', Q[q])
        title = kwargs.pop('title', None)
        
        if ax is None:
            plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.gca()
        
        if isinstance(axis, str): axis = 'xyz'.find(axis)
        
        data = None
        if project:
            data = np.sum(self.get(q, i=None), axis=axis)
            if log10: data = np.log10(data)
        else:
            data = self.get(q, i=i, log10=log10)
            if iSlice is None: iSlice = np.unravel_index(np.argmax(data), data.shape)[axis]
            data = getSlice(data, axis, iSlice)
        if len(np.shape(data))!=2: print(f"Requested data is not 2-dimensional (shape {np.shape(data)})."); return
        
        data = data[zoom[0]:zoom[1], zoom[2]:zoom[3]]
        extent = np.array([zoom[2], zoom[3], zoom[1], zoom[0]])*self.dx
        
        
        clims = kwargs.pop('clims', getColorLimits(data, climfactors))
        im = ax.imshow(data, extent=extent, cmap=cmap, vmin=clims[0], vmax=clims[1], **kwargs)
        axes = ['yz', 'zx', 'xy'][axis]
        ax.set(xlabel=rf"${axes[0]}$ [kpc]", ylabel=rf"${axes[1]}$ [kpc]")
        if title is None: title = r"($f_{15}$" + rf"$ = {self.f15}$, " + ("" if project else rf"${'xyz'[axis]} = {np.round(iSlice*self.dx,2)}$, ") + rf"$t = {np.round(self.t,3)}$)"
        ax.set_title(title)
        
        if colorbar is True:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(log10*r'$\log_{10}($' + Q[q] + log10*')' + f" {U.get(U.get(q), '')}")
        
        if save is True or filename!=Q[q]:
            plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return data
    
    
    
    ## Plots a cross-section of a quantity defined throughout the box.
    def slicePlot(self, q, axis=1, i=None, iSlice=None, ax=None, **kwargs):
        
        return self.plot2d(q, axis, i=i, iSlice=iSlice, ax=ax, **kwargs)

        
    
    ## Plots a projection of a quantity defined throughout the box.
    ## By "projection", I mean "simple sum" for now.
    def projectionPlot(self, q, axis=1, ax=None, **kwargs):
        
        return self.plot2d(q, axis, project=True, ax=ax, **kwargs)
    
    
    
    ## Animates cross-sections of a quantity through the box.
    def scan3d(self, q, axis=1, i=None, log10=False, **kwargs):
        
        dpi = kwargs.pop('dpi', 200)
        climfactors = kwargs.pop('climfactors', [0,1])
        fps = kwargs.pop('fps', 8)
        save = kwargs.pop('save', True)
        filename = kwargs.pop('filename', Q[q])
        
        data = self.get(q, i, log10)
        
        fig, ax = plt.subplots(dpi=dpi)
        clims = kwargs.pop('clims', getColorLimits(data, climfactors))
        colorbarmade = False
        
        def animate(j):
            
            nonlocal colorbarmade
            
            ax.clear()
            
            self.plot2d(q, axis, i=i, iSlice=j, log10=log10, ax=ax, clims=clims, colorbar=(not colorbarmade), **kwargs)
            colorbarmade = True
            
            return
        
        anim = ani.FuncAnimation(fig, animate, frames=self.N)
        writer = ani.PillowWriter(fps=fps)
        if save: anim.save(self.getPathAndName(filename, ".gif"), writer=writer)
        
        return data
    
    
    
    ## Computes the soliton density profile.
    ## Toggle 'plot' to plot and 'fit' to plot with the theoretical profile.
    def densityProfile(self, shells=20, normalize=True, fit=False, plot=True, ax=None, **kwargs):
        
        save = kwargs.pop('save', False)
        filename = kwargs.pop('filename', "Density Profile")
        
        rho = self.get('rho')
        
        i0 = np.array([int(self.N/2)]*3)
        iB = self.get('i0')
        rho = np.roll(rho, list(i0-iB), axis=np.arange(3))
        
        rmin, rmax = np.log10([self.dx, self.Lbox/2])
        r = np.logspace(rmin, rmax, shells)
        mids = 10**np.array([np.mean(np.log10([r[i], r[i+1]])) for i in range(shells-1)])
        
        x_ = np.linspace(0, self.Lbox, self.N)
        x, y, z = np.meshgrid(x_, x_, x_)
        x0, y0, z0 = x_[i0]
        dSq = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
        
        rho_in_shells = np.zeros(shells-1)
        for i in range(shells-1):
            iIn = np.array(np.where((r[i]**2 <= dSq) & (dSq < r[i+1]**2))).T
            rho_in_shells[i] = np.mean(rho[iIn[:,0], iIn[:,1], iIn[:,2]])
        
        rho0 = self.get('rho0')
        if normalize: 
            rho_in_shells = rho_in_shells/rho0
            mids = mids/self.get('rc')
        if plot or save or ax is not None:
            
            figsize = kwargs.pop('figsize', (6,4))
            dpi = kwargs.pop('dpi', 200)
            lims = kwargs.pop('lims', ([1e-1,1e2,1e-5,1] if normalize else [5e-2,5e1,1e2,1e10]))
            
            if ax is None: fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            label = rf'$\rho_0 = {np.round(rho0/1e10,1)}$' + r' [$10^{10}$ ' + U['density'] + ']'
            ax.loglog(mids, rho_in_shells, 'o-', label=label, **kwargs)
            ax.set_xlabel(r"$r$" + normalize*r"$/r_c$" + (not normalize)*f" {U['length']}")
            ax.set_ylabel(r"$\rho(r)$" + normalize*r"$/\rho_0$" + (not normalize)*f" {U['density']}")
            ax.set_title(r"($f_{15}$" + rf"$ = {self.f15}$, $t = {np.round(self.t,3)}$)")
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            
            if fit:
                r_fit = np.logspace(-1,2,100) if normalize else np.logspace(-2,1,100)
                rho_fit = solitonDensity(r_fit) if normalize else solitonDensity(r_fit, self.get('rho0'), self.get('rc'))
                ax.loglog(r_fit, rho_fit, c='k', label='Theoretical Non-SI Fit')
            
            ax.grid()
            plt.legend(loc=3)
            
        if save: plt.savefig(self.getPathAndName(filename, ".pdf"), transparent=True)
        
        return mids, rho_in_shells
    
    
    
    def getPathAndName(self, filename, ext):
        
        path = os.path.join(saveToParent, self.dir + " Images")
        if not os.path.isdir(path): path = saveToParent
        append = self.dir + " - " + str(self.num) + " - " + filename + ext
        path = os.path.join(path, append)
        
        return path



## Helper functions



## Gets the gradient of a quantity defined in the box.
def gradient(f, dx): 
    
    axes = np.arange(3)
    dfdx = (np.roll(f, [0,-1,0], axis=axes) - np.roll(f, [0,1,0], axis=axes))/(2*dx)
    dfdy = (np.roll(f, [-1,0,0], axis=axes) - np.roll(f, [1,0,0], axis=axes))/(2*dx)
    dfdz = (np.roll(f, [0,0,-1], axis=axes) - np.roll(f, [0,0,1], axis=axes))/(2*dx)
    
    return np.array([dfdx, dfdy, dfdz])



## Given 3-dimensional data, returns a 2-d cross-section along an axis.
## If sheet is None, returns slice containing largest value in data.
def getSlice(data, axis, iSlice=None):
    
    if iSlice is None: iSlice = np.unravel_index(np.argmax(data), data.shape)[axis]
    if axis==0: data = data[iSlice,:,:]
    if axis==1: data = data[:,iSlice,:]
    if axis==2: data = data[:,:,iSlice]
    
    return data



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



## Non-self-interacting soliton density profile function of radius.
## If either rho0 or rc is None, returns normalized functional values,
## otherwise returns dimensionful values. 
def solitonDensity(r, rho0=None, rc=None):
    
    return (1 + 0.091*r**2)**(-8) if (rho0 is None or rc is None) else rho0*(1 + 0.091*(r/rc)**2)**(-8)
    











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
    
        
        
        
        