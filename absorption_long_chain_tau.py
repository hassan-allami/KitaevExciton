import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from Kitaev_Spectrum import *

Compute = False  # decides to compute or load
# setting the parameters
N = 9
Parity = False  # False means Even parity
KE = KitaevQD(N, Parity, fixed_ne=False)
KO = KitaevQD(N, not Parity, fixed_ne=False)

T = -1
Delta = -1
Mu = 0
V = 10

N_T = 50
Tau = np.linspace(0, 0.3, N_T)

N_W = 1000
W_min = -10
W_max = 3
W = np.linspace(W_min, W_max, N_W)
sigma = 2.5e-2
eps = 1e-5
offset = 1e-10

Abs = np.zeros(((N + 3) // 2, N_T, N_W))
EE = np.zeros((N_T, 2 ** (N - 1) * N))
EO = np.zeros_like(EE)
AE = np.zeros_like(EE)
AO = np.zeros_like(EE)

if Compute:
    for i in range((N + 3) // 2):
        if i == 0:
            for t in range(N_T):
                t0 = time.time()
                EE[t], AE[t] = KE.absorption(T, Delta, Mu, Tau[t], V)
                EO[t], AO[t] = KO.absorption(T, Delta, Mu, Tau[t], V)
                t1 = time.time()
                print(f'{(i, t)} took {round(t1 - t0, 4)} seconds')
                # print(f'test: {len(EE)} == {2**(N-1)*N}')
            np.savetxt(f'data/Even_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}', EE)
            np.savetxt(f'data/Even_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}', AE)
            np.savetxt(f'data/Odd_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}', EO)
            np.savetxt(f'data/Odd_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}', AO)
        else:
            for t in range(N_T):
                t0 = time.time()
                EE[t], AE[t] = KE.absorption(T, Delta, Mu, Tau[t], V, full=False, site=i - 1)
                EO[t], AO[t] = KO.absorption(T, Delta, Mu, Tau[t], V, full=False, site=i - 1)
                t1 = time.time()
                print(f'{(i, t)} took {round(t1 - t0, 4)} seconds')
            np.savetxt(f'data/Even_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}', EE)
            np.savetxt(f'data/Even_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}', AE)
            np.savetxt(f'data/Odd_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}', EO)
            np.savetxt(f'data/Odd_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}', AO)

        for t in range(N_T):
            EE_t = EE[t][abs(AE[t]) > eps]
            AE_t = AE[t][abs(AE[t]) > eps]
            EO_t = EO[t][abs(AO[t]) > eps]
            AO_t = AO[t][abs(AO[t]) > eps]
            for a in range(len(AE_t)):
                Abs[i, t] += 0.5 * AE_t[a] * np.exp(-((W - EE_t[a]) / 2 / sigma) ** 2)
            for a in range(len(AO_t)):
                Abs[i, t] += 0.5 * AO_t[a] * np.exp(-((W - EO_t[a]) / 2 / sigma) ** 2)

else:
    for i in range((N + 3) // 2):
        if i == 0:
            EE = np.loadtxt(f'data/Even_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}')
            AE = np.loadtxt(f'data/Even_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}')
            EO = np.loadtxt(f'data/Odd_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}')
            AO = np.loadtxt(f'data/Odd_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}')
        else:
            EE = np.loadtxt(f'data/Even_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}')
            AE = np.loadtxt(f'data/Even_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}')
            EO = np.loadtxt(f'data/Odd_E_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}')
            AO = np.loadtxt(f'data/Odd_A_N{N}_Delta{Delta}_Mu{Mu}_V{V}_N_Tau{N_T}_{i}')

        for t in range(N_T):
            EE_t = EE[t][abs(AE[t]) > eps]
            AE_t = AE[t][abs(AE[t]) > eps]
            EO_t = EO[t][abs(AO[t]) > eps]
            AO_t = AO[t][abs(AO[t]) > eps]
            for a in range(len(AE_t)):
                Abs[i, t] += 0.5 * AE_t[a] * np.exp(-((W - EE_t[a]) / 2 / sigma) ** 2)
            for a in range(len(AO_t)):
                Abs[i, t] += 0.5 * AO_t[a] * np.exp(-((W - EO_t[a]) / 2 / sigma) ** 2)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting

fig, ax = plt.subplots(1, (N + 3) // 2, sharey=True, sharex=True, figsize=(12, 3.5))
fig.subplots_adjust(wspace=0.12, hspace=0., left=0.07, right=0.89, bottom=0.2, top=0.81)
fig.suptitle(r'$\overline{A}_{(i)} = \frac12 (A_{(i)}^{\rm even}+A_{(i)}^{\rm odd})$ ' +
             fr' , $N={N}$ , $V={V}|t|$', fontsize=18, y=0.95)
# for the 3 panel version
fig1, ax1 = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 5))
fig1.subplots_adjust(wspace=0.12, hspace=0, left=0.07, right=0.88, bottom=0.12, top=0.86)
fig1.suptitle(r'$\overline{A}_{(i)} = \frac12 (A_{(i)}^{\rm even}+A_{(i)}^{\rm odd})$ ' +
              fr' , $N={N}$ , $V={V}|t|$', fontsize=18, y=0.95)

for i in range((N + 3) // 2):
    im = ax[i].imshow(Abs[i] + offset, origin='lower',
                      extent=[W_min, W_max, Tau.min(), Tau.max()],
                      norm=LogNorm(vmin=1e-5, vmax=0.5),
                      interpolation='antialiased',
                      aspect='auto')
    ax[i].tick_params(labelsize=13)
    ax[i].set_xlabel('$(E-\eta)/|t|$', fontsize=15)
    # for the 3 panel version
    if i < 3:
        im1 = ax1[i].imshow(Abs[i] + offset, origin='lower',
                            extent=[W_min, W_max, Tau.min(), Tau.max()],
                            norm=LogNorm(vmin=1e-5, vmax=0.5),
                            interpolation='antialiased',
                            aspect='auto')
        ax1[i].tick_params(labelsize=13)
        ax1[i].set_xlabel('$(E-\eta)/|t|$', fontsize=15)
    if i == 0:
        # ax[i].set_title(r'$\overline{A}$', fontsize=16)
        ax[i].text(-3.5, 0.25, r'$\overline{A}$',
                   fontsize=16, ha='center', c='w')
        # for the 3 panel version
        ax1[i].text(-3.5, 0.25, r'$\overline{A}$',
                    fontsize=24, ha='center', c='w')
    else:
        # ax[i].set_title(r'$\overline{A}$'+f'$_{i}$', fontsize=16)
        ax[i].text(-3.5, 0.25, r'$\overline{A}$' + f'$_{i}$',
                   fontsize=16, ha='center', c='w')
        # for the 3 panel version
        if i < 3:
            ax1[i].text(-3.5, 0.25, r'$\overline{A}$' + f'$_{i}$',
                       fontsize=24, ha='center', c='w')

ax[0].set_ylabel(r'$\tau/|t|$', fontsize=15)
ax1[0].set_ylabel(r'$\tau/|t|$', fontsize=15)

cbar = fig.add_axes([0.91, 0.2, 0.02, 0.61])
fig.colorbar(im, cax=cbar)
cbar.tick_params(labelsize=13)
cbar.set_title(r'$\overline{A}_{(i)}$', fontsize=17, pad=13)
# for the 3 panel version
cbar = fig1.add_axes([0.91, 0.12, 0.02, 0.74])
fig1.colorbar(im1, cax=cbar)
cbar.tick_params(labelsize=13)
cbar.set_title(r'$\overline{A}_{(i)}$', fontsize=17, pad=13)

plt.show()
