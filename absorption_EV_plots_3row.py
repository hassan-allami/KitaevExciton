import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from Kitaev_Spectrum import *

# setting the parameters
N = 3
Parity = False  # False means Even parity
KE = KitaevQD(N, Parity)
KO = KitaevQD(N, not Parity)
T = -1
Delta = -1
Mu = 0
Tau = 0.1

N_W = 1000
W_min = -10
W_max = 5
W = np.linspace(W_min, W_max, N_W)
N_V = 100
V = np.linspace(0, 10, N_V)
sigma = 2.5e-2
eps = 1e-5
offset = 1e-10

# for the localized case
E0 = 2 - V
Ep = 1 - V/2 + np.sqrt(1 + (V/2)**2)
Em = 1 - V/2 - np.sqrt(1 + (V/2)**2)
A0 = ((N - 2) / 2 / N) * np.ones_like(V)
Ap = (1 - V / np.sqrt(4 + V**2)) / 2 / N
Am = (1 + V / np.sqrt(4 + V**2)) / 2 / N
E = [E0, Ep, Em]
A = [A0, Ap, Am]

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, ax = plt.subplots(3, N, sharey=True, sharex=True, figsize=(10, 10))
fig.subplots_adjust(wspace=0.12, hspace=0.15, left=0.11, right=0.88, bottom=0.11, top=0.89)
fig.suptitle(fr'$N={N}$, $\Delta = t < 0$,  $\mu = 0$, $\tau={Tau}|t|$',
             fontsize=16, y=0.98)
# for one row version
fig1, ax1 = plt.subplots(1, N, sharey=True, sharex=True, figsize=(10, 5))
fig1.subplots_adjust(wspace=0.12, hspace=0.15, left=0.07, right=0.88, bottom=0.11, top=0.85)
fig1.suptitle(r'$\overline{A}_{(i)} = \frac12 (A_{(i)}^{\rm even}+A_{(i)}^{\rm odd})$ ' +
              fr', $N={N}$, $\Delta = t < 0$,  $\mu = 0$, $\tau={Tau}|t|$', fontsize=16, y=0.98)

# for two rows version
fig2, ax2 = plt.subplots(2, N, sharey=True, sharex=True, figsize=(10, 5))
fig2.subplots_adjust(wspace=0.12, hspace=0.15, left=0.07, right=0.88, bottom=0.11, top=0.85)
fig2.suptitle(r'$\overline{A}_{(i)} = \frac12 (A_{(i)}^{\rm even}+A_{(i)}^{\rm odd})$ ' +
              fr', $N={N}$, $\Delta = t < 0$,  $\mu = 0$', fontsize=16, y=0.98)

Abs = np.zeros((3, N, N_V, N_W))

# for the localized case
Abs0 = np.zeros((N, N_V, N_W))
for n in range(N):
    for v in range(N_V):
        if n == 0:
            for i in range(len(A)):
                Abs0[n, v] += A[i][v] * np.exp(-((W - E[i][v]) / 2 / sigma) ** 2)
        elif n == 1:
            Abs0[n, v] += N / 2 * Ap[v] * np.exp(-((W - Ep[v]) / 2 / sigma) ** 2)
            Abs0[n, v] += N / 2 * Am[v] * np.exp(-((W - Em[v]) / 2 / sigma) ** 2)
        else:
            Abs0[n, v] += 0.5 * np.exp(-((W - E0[v]) / 2 / sigma) ** 2)

# for mobile hole
for m in range(3):
    for n in range(N):
        print(m, n)
        for v in range(N_V):
            if n % N == 0:
                EE, AE = KE.absorption(T, Delta, Mu, v=V[v], tau=Tau)
                EO, AO = KO.absorption(T, Delta, Mu, v=V[v], tau=Tau)
            else:
                EE, AE = KE.absorption(T, Delta, Mu, v=V[v], tau=Tau,
                                       full=False, site=(n % N) - 1)
                EO, AO = KO.absorption(T, Delta, Mu, v=V[v], tau=Tau,
                                       full=False, site=(n % N) - 1)
            EE = EE[abs(AE) > eps]
            AE = AE[abs(AE) > eps]
            EO = EO[abs(AO) > eps]
            AO = AO[abs(AO) > eps]

            if m == 0:
                for i in range(len(AE)):
                    Abs[m, n, v] += AE[i] * np.exp(-((W - EE[i]) / 2 / sigma) ** 2)
            elif m == 1:
                for i in range(len(AO)):
                    Abs[m, n, v] += AO[i] * np.exp(-((W - EO[i]) / 2 / sigma) ** 2)
            else:
                for i in range(len(AE)):
                    Abs[m, n, v] += 0.5 * AE[i] * np.exp(-((W - EE[i]) / 2 / sigma) ** 2)
                for i in range(len(AO)):
                    Abs[m, n, v] += 0.5 * AO[i] * np.exp(-((W - EO[i]) / 2 / sigma) ** 2)

        im = ax[m, n].imshow(Abs[m, n] + offset, origin='lower',
                             extent=[W_min, W_max, 0, 10],
                             norm=LogNorm(vmin=1e-5, vmax=0.5),
                             interpolation='antialiased',
                             aspect='auto')
        ax[m, n].tick_params(labelsize=13)
        if m == 2:
            ax[m, n].set_xlabel('$(E-\eta)/|t|$', fontsize=15)
            # plot the 1 row version
            im1 = ax1[n].imshow(Abs[m, n] + offset, origin='lower',
                                extent=[W_min, W_max, 0, 1],
                                norm=LogNorm(vmin=1e-5, vmax=0.5),
                                interpolation='antialiased',
                                aspect='auto')
            ax1[n].tick_params(labelsize=13)
            ax1[n].set_xlabel('$(E-\eta)/|t|$', fontsize=15)
            # plot the 2 rows version
            im2 = ax2[0, n].imshow(Abs[m, n] + offset, origin='lower',
                                   extent=[W_min, W_max, 0, 1],
                                   norm=LogNorm(vmin=1e-5, vmax=0.5),
                                   interpolation='antialiased',
                                   aspect='auto')
            ax2[0, n].tick_params(labelsize=13)
            ax2[0, n].text(-9, 0.1, r'$\tau=0.1|t|$',
                           fontsize=15, ha='left', c='w')
            im2 = ax2[1, n].imshow(Abs0[n] + offset, origin='lower',
                                   extent=[W_min, W_max, 0, 1],
                                   norm=LogNorm(vmin=1e-5, vmax=0.5),
                                   interpolation='antialiased',
                                   aspect='auto')
            ax2[1, n].tick_params(labelsize=13)
            ax2[1, n].set_xlabel('$(E-\eta)/|t|$', fontsize=15)
            ax2[1, n].text(-9, 0.1, r'$\tau=0$',
                           fontsize=15, ha='left', c='w')


ax[0, 0].set_title(r'full spectrum: $A$', fontsize=16)
ax[0, 1].set_title(r'end dot: $A_1$', fontsize=16)
ax[0, 2].set_title(r'middle dot: $A_2$', fontsize=16)

ax[0, 0].set_ylabel(r'even: $A_{(i)}^{\rm even}$' + '\n' + r'$V/|t|$',
                    fontsize=16)
ax[1, 0].set_ylabel(r'odd: $A_{(i)}^{\rm odd}$' + '\n' + r'$V/|t|$',
                    fontsize=16)
ax[2, 0].set_ylabel(r'$\frac12 (A_{(i)}^{\rm even}+A_{(i)}^{\rm odd}$)'
                    + '\n' + r'$\tau/|t|$',
                    fontsize=16)
# for 1 row version
ax1[0].set_title(r'full spectrum: $\overline{A}$', fontsize=16)
ax1[1].set_title(r'end dot: $\overline{A}_1$', fontsize=16)
ax1[2].set_title(r'middle dot: $\overline{A}_2$', fontsize=16)
ax1[0].set_ylabel(r'$V/|t|$', fontsize=15)
# for 2 rows version
ax2[0, 0].set_title(r'full spectrum: $\overline{A}$', fontsize=16)
ax2[0, 1].set_title(r'end dot: $\overline{A}_1$', fontsize=16)
ax2[0, 2].set_title(r'middle dot: $\overline{A}_2$', fontsize=16)
ax2[0, 0].set_ylabel(r'$V/|t|$', fontsize=15)
ax2[1, 0].set_ylabel(r'$V/|t|$', fontsize=15)

cbar = fig.add_axes([0.91, 0.11, 0.02, 0.78])
fig.colorbar(im, cax=cbar)
cbar.tick_params(labelsize=13)
cbar.set_title(r'$A_{(j)}$', fontsize=17, pad=13)
# for 1 row version
cbar = fig1.add_axes([0.91, 0.12, 0.02, 0.74])
fig1.colorbar(im1, cax=cbar)
cbar.tick_params(labelsize=13)
cbar.set_title(r'$\overline{A}_{(i)}$', fontsize=17, pad=13)
# for 2 row version
cbar = fig2.add_axes([0.91, 0.11, 0.02, 0.74])
fig2.colorbar(im2, cax=cbar)
cbar.tick_params(labelsize=13)
cbar.set_title(r'$\overline{A}_{(i)}$', fontsize=17, pad=13)

plt.show()
