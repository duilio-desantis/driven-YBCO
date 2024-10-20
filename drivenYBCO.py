import configparser
import ast
import os
import numpy as np
import itertools
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm

class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr  # Override to preserve case

def read_config(file_path):
    config = CaseSensitiveConfigParser()
    config.read(file_path)
    
    parameters = {}
    for section in config.sections():
        for key, value in config.items(section):
            try:
                parameters[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parameters[key] = value
    return parameters

def dataimport(filename):
    f = open(filename, "r")
    X = np.loadtxt(f, delimiter = ",")
    f.close()

    return X

# Some constants
omega = 1.2     # Driving frequency
tstart = 10.    # Driving start
nramp = 3.      # Driving width

period = (2.*np.pi)/omega
tramp = nramp*period

tpeak = tstart + 3.*tramp
tstop = tstart + 6.*tramp

# Driving envelope
def dshape(time):
    if time < tstart or time > tstop: return 0.
    else: return np.exp(-(time - tpeak)**2/(2.*(tramp**2)))

# Driving
def drive(time, V0):
    return V0*dshape(time)*np.sin(omega*(time - tstart))

# Driving_t
def drive_t(time, deltaT, V0):
    return (V0*dshape(time + deltaT)*np.sin(omega*(time + deltaT - tstart)) - V0*dshape(time - deltaT)*np.sin(omega*(time - deltaT - tstart)))/(2.*deltaT)      

# Vectorfield
def vectorfield(w, t, deltaX, deltaT, q, damp, V0):
    u = w[:: 2]
    u_t = w[1 :: 2]

    nX = len(u)

    # Create (u_tt[0], ..., u_tt[-1]):
    u_tt = []
    u_tt.extend([(2./(deltaX**2))*(u[1] - u[0]) - (2.*q/deltaX) + drive_t(t, deltaT, V0) - np.sin(u[0]) - damp*u_t[0]])
    u_tt.extend([(1./(deltaX**2))*(u[i + 1] - 2.*u[i] + u[i - 1]) + drive_t(t, deltaT, V0) - np.sin(u[i]) - damp*u_t[i] for i in range(1, nX - 1)])
    u_tt.extend([(2./(deltaX**2))*(u[-2] - u[-1]) + (2.*q/deltaX) + drive_t(t, deltaT, V0) - np.sin(u[-1]) - damp*u_t[-1]])
    
    # Create (u_t[0], u_tt[0], ..., u_t[-1], u_tt[-1]):
    vecfield = [z for z in itertools.chain.from_iterable(itertools.zip_longest(u_t, u_tt))]

    return vecfield

# Numerical integration and plots
def sg(L, Tscale, deltaX, deltaT, wX, wT, q, damp, V0, outdir):
    # Space & time grids
    Xleft = 0.
    Xright = L
    Tend = Tscale
    X = np.arange(Xleft, Xright, deltaX)
    T = np.arange(0., Tend, deltaT)

    nX = len(X)
    nT = len(T)

    idxstart = np.abs(T - tstart).argmin()
    
    # Initial conditions
    # u: initial phase | u_t: initial time derivative
    constq = q/(1. + np.exp(-X[-1]))
    u = [constq*(np.exp(X[i] - X[-1]) - np.exp(-X[i])) for i in range(nX)]
    u_t = nX*[0.0]
    
    u_x = np.zeros(nX)
    flux = np.zeros(nT)

    fname = os.path.join(outdir, "theta.txt")
    open(fname, 'w').close()
    f = open(fname, "a")

    fname_x = os.path.join(outdir, "grad-theta.txt")
    open(fname_x, 'w').close()
    f_x = open(fname_x, "a")

    # Second order finite difference for the space derivative
    # Internal cells -- centered
    for n in range(1, nX - 1):
        u_x[n] = (u[n + 1] - u[n - 1])/(2.*deltaX)
    # Left end -- forward
    u_x[0] = (-3.*u[0] + 4.*u[1] - u[2])/(2.*deltaX)
    # Right end -- backward
    u_x[-1] = (3.*u[-1] - 4.*u[-2] + u[-3])/(2.*deltaX)
    
    # Writes down the initial condition & its gradient
    if 0 % wT == 0:
        np.savetxt(f, [u[:: wX]], fmt = "%.10f", delimiter = ",")
        np.savetxt(f_x, [u_x[:: wX]], fmt = "%.10f", delimiter = ",")

    # Pack-up the initial conditions
    w0 = [z for z in itertools.chain.from_iterable(itertools.zip_longest(u, u_t))]

    for i in range(1, nT):

        print("t: {:.5f}".format(T[i]))
    
        # Call the ODE solver
        wsol = odeint(vectorfield, w0, [T[i - 1], T[i]], args = (deltaX, deltaT, q, damp, V0))[1]

        # Unpacks solution
        usol = wsol[:: 2]

        # Second order finite difference for the space derivative
        # Internal cells -- centered
        for n in range(1, nX - 1):
            u_x[n] = (usol[n + 1] - usol[n - 1])/(2.*deltaX)
        # Left end -- forward
        u_x[0] = (-3.*usol[0] + 4.*usol[1] - usol[2])/(2.*deltaX)
        # Right end -- backward
        u_x[-1] = (3.*usol[-1] - 4.*usol[-2] + usol[-3])/(2.*deltaX)
        
        flux[i] = usol[-1] - usol[0]

        # Writes down the current solution & its gradient
        if i % wT == 0:
            np.savetxt(f, [usol[:: wX]], fmt = "%.10f", delimiter = ",")
            np.savetxt(f_x, [u_x[:: wX]], fmt = "%.10f", delimiter = ",")

        w0 = wsol

    # Closes the files
    f.close()
    f_x.close()
    
    # Prepares the contour plot
    XC = np.arange(Xleft, Xright, deltaX)
    TC = np.arange(0., Tend, deltaT)

    TC = TC[:: wT]
    XC = XC[:: wX]

    uC = dataimport(fname)
    u_xC = dataimport(fname_x)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 6))
    
    # Plots the phase
    cs = ax[0].contourf(XC, TC, uC, 500)
    ax[0].set_xlabel(r"$ x $")
    ax[0].set_ylabel(r"$ t $")
    ax[0].set_title(r"$ \theta $")
    cbar = plt.colorbar(cs, ax = ax[0], spacing = 'proportional', orientation = 'horizontal')
    cbar.ax.locator_params(nbins = 5)

    # Plots the gradient
    cs = ax[1].contourf(XC, TC, u_xC, 500)
    ax[1].set_xlabel(r"$ x $")
    ax[1].set_ylabel(r"$ t $")
    ax[1].set_title(r"$ \partial_{{x}} \theta $")
    cbar = plt.colorbar(cs, ax = ax[1], spacing = 'proportional', orientation = 'horizontal')
    cbar.ax.locator_params(nbins = 5)

    fig.suptitle(r"$ \alpha = {:.3f}, \; V_0 = {:.2f}, \; \omega = {:.2f}, \; q = {:.6f} $".format(damp, V0, omega, q))
    fig.savefig(os.path.join(outdir, "contour_theta.png"), format = 'png', bbox_inches = 'tight', dpi = 500)
    plt.close(fig)
    
    # Plots the induced magnetic flux vs. time
    fig, ax = plt.subplots()
    ax.plot(T[idxstart :], flux[idxstart :] - flux[idxstart])
    ax.set_ylabel(r"$ \Phi(t) - \Phi(0) $")
    ax.set_xlabel(r"$ t $")
    ax.set_title(r"$ \alpha = {:.3f}, \; V_0 = {:.2f}, \; \omega = {:.2f}, \; q = {:.6f} $".format(damp, V0, omega, q))
    fig.savefig(os.path.join(outdir, "flux.png"), format = 'png', bbox_inches = 'tight', dpi = 500)
    plt.close(fig)
    
    return flux

def main():
    # Parameters
    cfile = 'config.txt'
    outfile = 'flux.txt'
    
    # Read config file
    params = read_config(cfile)
    
    # Parameter-based output folder
    included_params = ['L', 'q', 'damp', 'V0']
    outdir = "_".join([f"{key}={params[key]}" for key in included_params])
    os.makedirs(outdir, exist_ok = True)
    
    # Runs simulation
    flux = sg(**params, outdir = outdir)

    # Saves induced flux to file
    outfile_path = os.path.join(outdir, outfile)
    np.savetxt(outfile_path, np.array([flux]))

if __name__ == "__main__":
    main()