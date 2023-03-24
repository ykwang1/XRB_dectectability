import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import poisson, bernoulli, binom

class XRBsimulator():
    SIM_POSITIONS = '../data/sample_xrb_positions_10M.csv.gz'
    COV_DIR = '../data/CovMaps_20220901.txt'
    WATCHDOG_DIR = '../data/watchdog.txt'
    NOBS_DIR = '../data/n_obs_partnership_public.csv.gz'
    FIELDS_DIR = "../data/fields/"

    QLUM = 10 ** 31 * u.erg
    OLUM = 10 ** 39 * u.erg
    ZTF_LIM = 20.5
    DC_override = False

    OABSMAG = 0

    SURVEYS = ['ROSAT', 'Swift', 'Chandra', 'XMM', 'XMMSL']
    SKYCOV = {'ROSAT':1, 'Swift': .091, 'Chandra': .013, 'XMM': .029, 'XMMSL': .84}
    PCT_DIS = {'ROSAT': {'Q': 2.5e-05, 'O': 0.99999}, 'Swift': {'Q': 0.001385, 'O': 1.0}, 'Chandra': {'Q': 0.015561, 'O': 0.992085}, 'XMM': {'Q': 0.177291, 'O': 1.0}, 'XMMSL': {'Q': 0.001276, 'O': 0.373686}}
    # PCT_DIS = {'ROSAT': {'Q': 2.4e-05, 'O': 0.99999}, 'Swift': {'Q': 0.002506, 'O': 1.0}, 'Chandra': {'Q': 0.032189, 'O': 0.999978}, 'XMM': {'Q': 0.528361, 'O': 1.0}, 'XMMSL': {'Q': 0.002637, 'O': 0.998595}}

    def __init__(self):
        m_ztf = 20.5  # ZTF limiting mag
        del_m = 6  # difference in outburst and quescient mag (optical)
        R_V = 3.1  #  E (B-V) to A_v

        self.df = pd.read_csv(self.SIM_POSITIONS, header=None, names=['ra', 'dec', 'd_kpc', 'E(B-V)'])

        self.df['distmod'] = 5 * np.log10(self.df["d_kpc"] * 1.e3) - 5.0  # Calculate distance modulus for all XRB
        self.df['A_V'] = R_V * self.df['E(B-V)']   # Calculate extinction  https://ned.ipac.caltech.edu/level5/Sept07/Li2/Li2.html
        self.df['nH'] = 2.21 * 10 ** 21 * self.df['A_V']   # column density in cm ** -2 https://arxiv.org/pdf/0903.2057.pdf

        self.sim = pd.DataFrame(index=self.df.index)

        self._field_cut_data()

        self.wd = self._init_watchdog()

        self.df['wd'] = np.random.choice(np.arange(len(self.wd)), size=len(self.df))

        self.prob = None

    def _galactic_latitude(self, ra, dec):
        # l_ref = 33.012 # deg
        ra_ref = 282.25 # deg
        g = 62.6 # deg
        b =  np.arcsin(np.sin(np.deg2rad(dec)) * np.cos(np.deg2rad(g)) - \
                       np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(g)) * np.sin(np.deg2rad(ra) - np.deg2rad(ra_ref)))
        return np.rad2deg(b)


    def _field_cut_data(self):
        # read in ZTF obs per field
        nobs = pd.read_csv(self.NOBS_DIR)
        nobs.columns = ['fieldID', 'filter', 'nobs']

        # read in field of each simulated object (index)
        fields = []
        for ii in range(0, 100):
            fields.append(pd.read_csv(self.FIELDS_DIR+f'{ii}.csv'))

        fields = pd.concat(fields)
        fields.columns = ['index', 'ccd', 'qid', 'rcid', 'field']

        # merge number of obs with df of fields of each object
        fields = fields.merge(nobs, left_on='field', right_on='fieldID')
        del nobs

        # read in coverage maps
        cov_maps = pd.read_csv(self.COV_DIR, delimiter='|', header=0, skiprows=[1], skipfooter=1, engine='python')
        cov_maps.columns = [x.strip() for x in cov_maps.columns]
        cov_maps['utobsdate_earliest'] = pd.to_datetime(cov_maps['utobsdate_earliest'])
        convert_fid = lambda x: 'g' if x==1 else ('r' if x==2 else 'i')
        cov_maps['filter'] = [convert_fid(fid) for fid in cov_maps['fid']]

        # join with coverage data and earliest obs data
        obs_df = fields.merge(cov_maps[['utobsdate_earliest', 'maglimcat', 'rcid', 'field', 'filter']], on=['rcid', 'field', 'filter'], how='left')
        obs_df.drop(['ccd', 'qid', 'rcid', 'fieldID', 'filter'], axis=1, inplace=True)

        # aggregate sum of observations and earliest observation for each object
        obs_df = obs_df.groupby('index')[['nobs', 'utobsdate_earliest']].agg({'nobs':sum, 'utobsdate_earliest':min})

        self.df = pd.merge(self.df, obs_df, left_index=True, right_index=True, how='left')


    def _est_det(self, qpct, opct, skycov, dc=.027):
        if self.DC_override:
            dc = 0.027
        return(skycov * ((qpct * (1-dc) + opct * dc)))


    def _init_watchdog(self):
        # Use data from watchdog database of XRBs
        wd = pd.read_csv(self.WATCHDOG_DIR, delim_whitespace=True)

        # turn recurrence times to integers
        wd['trec'] = wd.trecur.str.strip('>').astype(int)

        # lower bound flag
        wd['lb'] = [1 if x == '>' else 0 for x in wd.trecur.str[0]]

        # estimate recurrence time and duty cycle by doubling trec for those with lower bounds
        wd['trec_est'] = wd['trec'] * (wd['lb'] + 1) - 1
        wd['dc_est'] = wd['tout']/(wd['tout'] + wd['trec_est'])

        # find mean outburst duraction
        wd['tout_mean'] = wd['tout'] / wd['Total']

        # change duty cycle from pct to number
        # wd['dc_est'] /= 100
        wd['dc'] /= 100

        # calculate pct chance each object is caught by surveys
        for survey in self.SURVEYS:
            wd[f'{survey}'] = [self._est_det(self.PCT_DIS[survey]['Q'], self.PCT_DIS[survey]['O'], self.SKYCOV[survey], dc)for dc in wd['dc_est']]

        # calculate probability of observing each XRB
        pobs = []
        N = 1505 # days
        rec = 'trec_est'
        for ii in range(len(wd)):
            trec = wd.loc[ii, rec]
            tout = wd.loc[ii, 'tout_mean']
            Nob = (N + tout) / trec
            pobs.append(1 - poisson.cdf(1, Nob))

        wd['pobs'] = pobs

        return wd


    def _make_draw(self, name, pct):
        draw_sims = pd.Series(np.random.random(size=int(1e7)), name=name)
        hits = draw_sims < pct
        hits.name = name
        return hits


    def _in_cat_sim(self):
        data = pd.DataFrame(self.df['wd'])
        for survey in self.SURVEYS:
            data = data.merge(self._make_draw(survey, self.wd.loc[data['wd'], survey].reset_index(drop=True)), left_index=True, right_index=True)

        self.sim['in_cat'] = data[self.SURVEYS].sum(axis=1).astype('bool')
        print(f"pct in xray catalog: {self.sim['in_cat'].mean()}")


    def _in_ob_sim(self):
        data = pd.DataFrame(self.df['wd'])
        self.sim['in_ob'] = self._make_draw('in_ob', self.wd.loc[self.df['wd'], 'pobs'].reset_index(drop=True))

        print(f"pct in outburst: {self.sim['in_ob'].mean()}")



    def _optical_bright_enough(self):
        # Calculate distance modulus
        self.sim['bright_enough'] = (self.df['distmod'] + self.df['A_V'] + self.OABSMAG) < self.ZTF_LIM
        print(f"pct above ZTF maglim: {self.sim['bright_enough'].mean()}")


    def _field_observed_sim(self):
        self.sim['in_ZTF'] = self._make_draw('in_ZTF', (~self.df['utobsdate_earliest'].isna()).mean())
        print(f"pct in ZTF fields: {self.sim['in_ZTF'].mean()}")

    def _in_b_15(self):
        self.sim['in_b_15'] = (abs(self._galactic_latitude(self.df['ra'], self.df['dec'])) < 15)
        print(f"pct in galactic plane: {self.sim['in_b_15'].mean()}")

    def _trial(self, N, prob, ntrials=10000):
        nsuccess = (binom.rvs(N, prob, size=ntrials) == 5).sum()
        return nsuccess / ntrials

    def create_sim(self):
        self._field_observed_sim()
        self._in_ob_sim()
        self._in_cat_sim()
        self._in_b_15()
        self._optical_bright_enough()

        self.sim['det'] = self.sim[['in_ob', 'in_cat', 'in_ZTF', 'in_b_15', 'bright_enough']].mean(axis=1)
        self.sim['det'] = self.sim['det'] >= 1

        print(f"overall pct detected {self.sim['det'].mean()}")
        self.prob = self.sim['det'].mean()

    def estimate_pdf(self):
        if self.prob is None:
            return
        xs = np.linspace(10, 1000, 500)
        ps = []
        for x in xs:
            ps.append(self._trial(int(x), self.prob))
        plt.plot(xs, ps, label=f'MLE={int(5/self.prob)}')
        plt.title("Probability Distribution of Number of XRBs")
        plt.xlabel('Number of XRBs')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    test = XRBsimulator()
    test.create_sim()
    test.estimate_pdf()
    import pdb; pdb.set_trace()
