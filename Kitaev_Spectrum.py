import numpy as np
import scipy.sparse as spr
from itertools import product
from functools import lru_cache
import matplotlib.pyplot as plt
import time


class KitaevQD:
    """
    This class generates the Hamiltonian with holes
    and it computes the spectral function of the electrons
    as well as absorption of the system
    """

    def __init__(self, length: int, parity: bool,
                 fixed_ne: bool = False, ne: int = 0,
                 nh: int = 0,
                 ):
        """
        parity: if False means "Even" and if True means "Odd"
        fixed_ne: if False nothing happens and
        if true it uses ne
        """
        self.length = length
        self.nh = nh
        self.parity = parity
        self.fixed_ne = fixed_ne
        self.ne = ne

    @lru_cache  # this memoize
    def configs(self):
        """
        generates configurations as a dictionary
        labels are tuples and indexes are integers
        the first N digits belong to electrons and
        the second to the holes
        """
        confs = list(product([0, 1], repeat=2 * self.length))
        confs = [tup for tup in confs if sum(tup[self.length:]) == self.nh]
        if not self.fixed_ne:
            if self.parity:
                confs = [tup for tup in confs if \
                         sum(tup[: self.length]) % 2 == 1]
            else:
                confs = [tup for tup in confs if \
                         sum(tup[: self.length]) % 2 == 0]
        else:
            confs = [tup for tup in confs if sum(tup[: self.length]) == self.ne]
        return {tpl: idx for idx, tpl in enumerate(confs)}

    @lru_cache  # this memoize
    def ne_i(self, site: int):
        """
        generates the operator ne_i = c_i+ * c_i
        at site i
        """
        confs = self.configs()
        dim = len(confs)
        dat = []
        row = []
        col = []
        for c in confs:
            if c[site] == 1:
                dat.append(1)
                row.append(confs[c])
                col.append(confs[c])
        return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))

    @lru_cache  # this memoize
    def nh_i(self, site: int):
        """
        generates the operator nh_i = h_i+ * h_i
        at site i
        """
        confs = self.configs()
        dim = len(confs)
        dat = []
        row = []
        col = []
        for c in confs:
            if c[site + self.length] == 1:
                dat.append(1)
                row.append(confs[c])
                col.append(confs[c])
        return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))

    @lru_cache  # this memoize
    def nei_nhi(self, site: int):
        """
        this generates the density-density local interaction
        ne_i * nh_i
        """
        return self.ne_i(site) @ self.nh_i(site)

    @lru_cache  # this memoize
    def hop_e(self, sites: (int, int)):
        """
        generates the operator c_i+ * c_j
        where sites = (i, j)
        """
        if sites[0] == sites[1]:
            return self.ne_i(site=sites[0])
        elif sites[0] < sites[1]:
            confs = self.configs()
            dim = len(confs)
            dat = []
            row = []
            col = []
            for c in confs:
                if c[sites[0]] == 0 and c[sites[1]] == 1:
                    # print(c)
                    phase = sum(c[sites[0]: sites[1]]) % 2
                    new_c = list(c)
                    new_c[sites[0]] = 1
                    new_c[sites[1]] = 0
                    new_c = tuple(new_c)
                    dat.append((-1) ** phase)
                    row.append(confs[new_c])
                    col.append(confs[c])
            return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))
        else:
            return self.hop_e(sites=sites[::-1]).T

    @lru_cache  # this memoize
    def hop_h(self, sites: (int, int)):
        """
        generates the operator h_i+ * h_j
        where sites = (i, j)
        """
        if sites[0] == sites[1]:
            return self.nh_i(site=sites[0])
        elif sites[0] < sites[1]:
            confs = self.configs()
            dim = len(confs)
            dat = []
            row = []
            col = []
            for c in confs:
                if c[sites[0] + self.length] == 0 and c[sites[1] + self.length] == 1:
                    # print(c)
                    phase = sum(c[sites[0] + self.length: sites[1] + self.length]) % 2
                    new_c = list(c)
                    new_c[sites[0] + self.length] = 1
                    new_c[sites[1] + self.length] = 0
                    new_c = tuple(new_c)
                    dat.append((-1) ** phase)
                    row.append(confs[new_c])
                    col.append(confs[c])
            return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))
        else:
            return self.hop_h(sites=sites[::-1]).T

    @lru_cache  # this memoize
    def pair(self, sites: (int, int)):
        """
        generates the operator c_i+ * c_j+ + c_j * c_i
        where sites = (i, j)
        """
        confs = self.configs()
        dim = len(confs)
        if sites[0] == sites[1]:
            return spr.coo_matrix((dim, dim))
        elif sites[0] < sites[1]:
            dat = []
            row = []
            col = []
            for c in confs:
                if c[sites[0]] == 0 and c[sites[1]] == 0:
                    phase = sum(c[sites[0]: sites[1]]) % 2
                    new_c = list(c)
                    new_c[sites[0]] = 1
                    new_c[sites[1]] = 1
                    new_c = tuple(new_c)
                    dat.append((-1) ** phase)
                    row.append(confs[new_c])
                    col.append(confs[c])
            mat = spr.coo_matrix((dat, (row, col)), shape=(dim, dim))
            mat += mat.T
            return mat
        else:
            return -self.pair(sites=sites[::-1])

    @lru_cache  # this memoize
    def cj_plus(self, site):
        """
        This generates the operator c_j+
        If fixed_ne is False then
        it connects parity to !parity and
        If fixed_ne is True then
        it connects ne to ne + 1
        """
        if not self.fixed_ne:
            k1 = KitaevQD(self.length, not self.parity,
                          self.fixed_ne, self.ne, self.nh)
        else:
            k1 = KitaevQD(self.length, self.parity,
                          self.fixed_ne, 1 + self.ne, self.nh)

        conf0 = self.configs()
        conf1 = k1.configs()
        d0 = len(conf0)
        d1 = len(conf1)
        dat = []
        row = []
        col = []
        for c in conf0:
            if c[site] == 0:
                # print(c)
                phase = sum(c[: site]) % 2
                new_c = list(c)
                new_c[site] = 1
                new_c = tuple(new_c)
                # print(new_c)
                dat.append((-1) ** phase)
                col.append(conf0[c])
                row.append(conf1[new_c])
        return spr.coo_matrix((dat, (row, col)), shape=(d1, d0))

    @lru_cache  # this memoize
    def hj_plus(self, site):
        """
        This generates the operator h_j+
        it connects nh to nh + 1 subspace
        for electrons stays either in the same parity
        or the same ne
        depending on the value of fixed_ne
        """
        k1 = KitaevQD(self.length, self.parity,
                      self.fixed_ne, self.ne, 1 + self.nh)
        conf0 = self.configs()
        conf1 = k1.configs()
        d0 = len(conf0)
        d1 = len(conf1)
        dat = []
        row = []
        col = []
        for c in conf0:
            if c[site + self.length] == 0:
                # print(c)
                phase = sum(c[: site + self.length]) % 2
                new_c = list(c)
                new_c[site + self.length] = 1
                new_c = tuple(new_c)
                # print(new_c)
                dat.append((-1) ** phase)
                col.append(conf0[c])
                row.append(conf1[new_c])
        return spr.coo_matrix((dat, (row, col)), shape=(d1, d0))

    @lru_cache  # this memoize
    def p_j(self, site):
        """
        This generates the operator c_j+ * h_j+
        it connects nh to nh + 1 subspace
        If fixed_ne is False then
        it connects parity to !parity and
        If fixed_ne is True then
        it connects ne to ne + 1
        """
        k1 = KitaevQD(self.length, self.parity,
                      self.fixed_ne, self.ne, 1 + self.nh)
        hj = self.hj_plus(site)
        cj = k1.cj_plus(site)
        return cj @ hj

    @lru_cache  # this memoize
    def t_term(self):
        """
        this generates the electron hopping term
        Sum c_{i+1}+ * c_{i} + h.c.
        """
        dim = len(self.configs())
        t_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length - 1):
            term = self.hop_e(sites=(i + 1, i))
            term += term.T
            # print(term)
            t_mat += term
        return t_mat

    @lru_cache  # this memoize
    def tau_term(self):
        """
        this generates the hole hopping term
        -Sum h_{i+1}+ * h_{i} + h.c.
        """
        dim = len(self.configs())
        t_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length - 1):
            term = self.hop_h(sites=(i + 1, i))
            term += term.T
            # print(term)
            t_mat -= term
        return t_mat

    @lru_cache  # this memoize
    def delta_term(self):
        """
        this generates the electron pairing term
        Sum c_{i+1}+ * c_{i}+ + h.c.
        """
        dim = len(self.configs())
        d_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length - 1):
            d_mat += self.pair(sites=(i + 1, i))
        return d_mat

    @lru_cache  # this memoize
    def mu_term(self):
        """
        this generates the chemical potential term
        -ne_tot
        """
        dim = len(self.configs())
        u_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length):
            u_mat -= self.ne_i(i)
        return u_mat

    @lru_cache  # this memoize
    def gap_term(self):
        """
        this generates the gap term
        +nh_tot
        """
        dim = len(self.configs())
        g_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length):
            g_mat += self.nh_i(i)
        return g_mat

    @lru_cache  # this memoize
    def int_term(self):
        """
        this generates the e-h interaction term
        - Sum ne_i * nh_i
        """
        dim = len(self.configs())
        v_mat = spr.coo_matrix((dim, dim))
        for i in range(self.length):
            v_mat -= self.nei_nhi(i)
        return v_mat

    @lru_cache  # this memoize
    def ham(self, t, delta, mu, tau=0., v=0., eta=0.):
        """
        this puts together all terms of the Hamiltonian
        when fixed_ne is True it drops the pairing term
        """
        if not self.fixed_ne:
            return (t * self.t_term() + delta * self.delta_term() +
                    mu * self.mu_term() + tau * self.tau_term() +
                    v * self.int_term() + eta * self.gap_term())
        else:
            return (t * self.t_term() +
                    mu * self.mu_term() + tau * self.tau_term() +
                    v * self.int_term() + eta * self.gap_term())

    @lru_cache  # this memoize
    def spectral_fun(self, site, t, delta, mu, tau=0., v=0., eta=0.):
        """
        This generates electronic spectral function
        by "site" with an electron.
        If fixed_ne is False then
        c_j+ is applied on the lowest state of the parity subspace
        If fixed_ne is True then
        c_j+ is applied on the lowest state of the ne subspace
        It returns the locations and the heights of the peaks
        """
        if not self.fixed_ne:
            k1 = KitaevQD(self.length, not self.parity,
                          self.fixed_ne, self.ne, self.nh)
        else:
            k1 = KitaevQD(self.length, self.parity,
                          self.fixed_ne, 1 + self.ne, self.nh)
        cp = self.cj_plus(site)
        h0 = self.ham(t, delta, mu, tau, v, eta)
        h1 = k1.ham(t, delta, mu, tau, v, eta)
        e0, p0 = np.linalg.eigh(h0.toarray())
        e1, p1 = np.linalg.eigh(h1.toarray())
        e = e1 - min(e0)
        a = abs(p1.T @ cp @ p0[:, 0]) ** 2
        return e, a

    @lru_cache  # this memoize
    def absorption(self, t, delta, mu, tau=0., v=0., eta=0.,
                   full=True, site=0):
        """
        This generates absorption spectrum.
        If full is True it uses the full polarization operator
        P = Sum(P_j) / sqrt(N)
        If full is False it uses site for P_j = c_j+ * h_j+
        where j = site
        The output energy series is E - eta and
        since eta only causes a shift the output is independent of eta
        """
        if full:
            p = self.p_j(self.length - 1)
            for j in range(self.length - 1):
                p += self.p_j(j)
            p = p / np.sqrt(self.length)
        else:
            p = self.p_j(site)

        if not self.fixed_ne:
            k1 = KitaevQD(self.length, not self.parity,
                          self.fixed_ne, self.ne, 1 + self.nh)
        else:
            k1 = KitaevQD(self.length, self.parity,
                          self.fixed_ne, 1 + self.ne, 1 + self.nh)

        h0 = self.ham(t, delta, mu, tau, v, eta)
        h1 = k1.ham(t, delta, mu, tau, v, eta)
        e0, p0 = np.linalg.eigh(h0.toarray())
        e1, p1 = np.linalg.eigh(h1.toarray())
        e = e1 - min(e0) - eta
        a = abs(p1.T @ p @ p0[:, 0]) ** 2
        return e, a


if __name__ == "__main__":
    N = 5
    Parity = False
    T = 1
    Delta = np.random.random()
    Delta = -1
    U = 0
    V = 5
    Tau = 0.1
    Em = -V + 2 * abs(T)
    E0 = -V / 2 + abs(T) - np.sqrt(T ** 2 + (V / 2) ** 2)
    E1 = -V / 2 + abs(T) + np.sqrt(T ** 2 + (V / 2) ** 2)
    Am = (N - 2) / 2 / N
    A0 = (1 + V / np.sqrt(4 * T ** 2 + V ** 2)) / 2 / N
    A1 = (1 - V / np.sqrt(4 * T ** 2 + V ** 2)) / 2 / N

    K = KitaevQD(length=N, parity=Parity, nh=0, fixed_ne=False, ne=0)
    K1 = KitaevQD(length=N, parity=not Parity, nh=0, fixed_ne=False, ne=0)

    print('\n'.join(f'{key}: {value}' for key, value in K.configs().items()))

    # E, A = K.spectral_fun(0, T, Delta, U)
    #
    # print('\n', 'E =', E.round(3), '\n')
    # print('\n', 'A =', A.round(3), '\n')
    # print('sum test:', A.sum())

    # print('\n', K.hj_plus(1).toarray(), '\n')
    # print('\n', K.hj_plus(1).shape, '\n')

    E, A = K.absorption(T, Delta, U, tau=Tau, v=V, full=False, site=0)
    EP, AP = K1.absorption(T, Delta, U, tau=Tau, v=V, full=False, site=0)
    EL, AL = K.absorption(T, Delta, U, tau=Tau, v=V, full=False, site=0)
    EM, AM = K.absorption(T, Delta, U, tau=Tau, v=V, full=False, site=1)

    print('\n', 'E =', E.round(3), '\n')
    print('\n', 'A =', A.round(3), '\n')
    print('sum test Even:', A.sum())
    print('sum test Odd:', AP.sum())
    print('sum test together:', A.sum() + AP.sum())

    eps = 1e-6
    print(len(A[abs(A) > eps]))
    print(len(np.unique(np.round(E[abs(A) > eps], 6))))

    ET, AT = K.absorption(T, 0.1, U, Tau, V, 0, full=True, site=1)
    # print(len(np.unique(ET.round(5))))
    # E_T = np.asarray(
    #     [-9.022248419781258, -9.010595629748021, -8.536666845245717, -7.988229241866264, -7.350117269411916,
    #      -7.3325268157993015, 0.7804690527111826, 0.7862600622838213, 1.7067696176398748, 2.0179974067244113,
    #      2.7875356387829826, 2.795099827648043])
    # A_T = np.asarray(
    #     [0.3130342039786633, 1.111135339623347e-26, 0.02102548490958582, 0.15968348365610635, 6.626842468911301e-29,
    #      0.01444537482921768, 5.172458656555295e-28, 0.003304656191696746, 9.735878939790123e-34,
    #      5.0538215190295395e-05, 1.5730032820782749e-31, 8.871140702135867e-05])
    # print('E result:', ET.round(5))
    # print('A result:', AT.round(5))
    # print('E test:', (ET - E_T).round(5))
    # print('A test:', (AT - A_T).round(5))

    E = E[abs(A) > eps]
    A = A[abs(A) > eps]
    EP = EP[abs(AP) > eps]
    AP = AP[abs(AP) > eps]
    EL = EL[abs(AL) > eps]
    AL = AL[abs(AL) > eps]
    EM = EM[abs(AM) > eps]
    AM = AM[abs(AM) > eps]
    ET = ET[abs(AT) > eps]
    AT = AT[abs(AT) > eps]
    N_mesh = 5000
    sigma = 1.5e-2
    W = np.linspace(min(E) - 4 * sigma, max(E) + 4 * sigma, N_mesh)
    WL = np.linspace(min(EL) - 4 * sigma, max(EL) + 4 * sigma, N_mesh)
    WM = np.linspace(min(EM) - 4 * sigma, max(EM) + 4 * sigma, N_mesh)
    WT = np.linspace(min(ET) - 4 * sigma, max(ET) + 4 * sigma, N_mesh)
    WP = np.linspace(min(EP) - 4 * sigma, max(EP) + 4 * sigma, N_mesh)
    Abs = np.zeros(N_mesh)
    AbsL = np.zeros(N_mesh)
    AbsM = np.zeros(N_mesh)
    AbsT = np.zeros(N_mesh)
    AbsP = np.zeros(N_mesh)
    for i in range(len(A)):
        Abs += A[i] * np.exp(-((W - E[i]) / sigma) ** 2)
    for i in range(len(AL)):
        AbsL += AL[i] * np.exp(-((WL - EL[i]) / sigma) ** 2)
    for i in range(len(AM)):
        AbsM += AM[i] * np.exp(-((WM - EM[i]) / sigma) ** 2)
    for i in range(len(AT)):
        AbsT += AT[i] * np.exp(-((WT - ET[i]) / sigma) ** 2)
    for i in range(len(AP)):
        AbsP += AP[i] * np.exp(-((WP - EP[i]) / sigma) ** 2)

    base = 1e-10
    plt.plot(W, Abs + base)
    plt.yscale('log')
    plt.ylim(1e-6, 1 / 2)
    plt.axvline(x=Em, ls=':', c='grey')
    plt.axvline(x=E0, ls=':', c='grey')
    plt.axvline(x=E1, ls=':', c='grey')
    plt.axhline(y=Am + base, ls=':', c='grey')
    plt.axhline(y=A0 + base, ls=':', c='grey')
    plt.axhline(y=A1 + base, ls=':', c='grey')

    plt.plot(WP, AbsP + base, '--')
    # plt.plot(WL, AbsL + base, ':')
    # plt.plot(WM, AbsM + base, ':')
    # plt.plot(WT, AbsT + base, '--')

    # plt.figure()
    # y = np.random.random((3, 5))
    # x_min = np.arange(0, 5, 1)
    # x_max = np.arange(0.5, 5.5, 1)
    # x_min = np.kron(np.ones((3, 1)), x_min)
    # x_max = np.kron(np.ones((3, 1)), x_max)
    # print(y)
    # # print(np.kron(np.ones((3, 1)), x_min))
    # print(x_max)
    # plt.hlines(y, x_min, x_max)

    # plt.show()

