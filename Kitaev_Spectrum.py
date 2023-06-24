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
