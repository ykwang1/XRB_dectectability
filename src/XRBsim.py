import numpy as np
import pandas as pd
import astropy.units as u
from scipy.stats import poisson, bernoulli

class XRBsimulator():
    SIM_POSITIONS = '../data/sample_xrb_positions_10M.csv.gz'
    COV_DIR = '../data/CovMaps_20220901.txt'
    WATCHDOG_DIR = '../data/watchdog.txt'
    NOBS_DIR = '../data/n_obs_partnership_public.csv.gz'
    FIELDS_DIR = "../data/fields/"

    QLUM = 10 ** 31 * u.erg
    OLUM = 10 ** 39 * u.erg
    ZTF_LIM = 20.5
    DC_override = True

    OABSMAG = 0

    SURVEYS = ['ROSAT', 'Swift', 'Chandra', 'XMM', 'XMMSL']
    SKYCOV = {'ROSAT':1, 'Swift': .091, 'Chandra': .013, 'XMM': .029, 'XMMSL': .84}
    PCT_DIS = {'ROSAT': {'Q': 2.4e-05, 'O': 0.99999}, 'Swift': {'Q': 0.002506, 'O': 1.0}, 'Chandra': {'Q': 0.032189, 'O': 0.999978}, 'XMM': {'Q': 0.528361, 'O': 1.0}, 'XMMSL': {'Q': 0.002637, 'O': 0.998595}}

    def __init__(self):
        m_ztf = 20.5  # ZTF limiting mag
        del_m = 6  # difference in outburst and quescient mag (optical)
        R_V = 3.1  #  E (B-V) to A_v

        self.df = pd.read_csv(self.SIM_POSITIONS, header=None, names=['ra', 'dec', 'd_kpc', 'E(B-V)'])

        self.df['distmod'] = 5 * np.log10(self.df["d_kpc"] * 1.e3) - 5.0  # Calculate distance modulus for all XRB
        self.df['A_V'] = R_V * self.df['E(B-V)']   # Calculate extinction
        self.df['nH'] = 2.21 * 10 ** 21 * self.df['A_V']   # column density in cm ** -2

        self.field_cut_data()

        self.wd = self.init_watchdog()
        self.df['wd'] = np.random.choice(np.arange(len(self.wd)), size=len(self.df))

        self.field_observed_sim()

    def galactic_latitude(self, ra, dec):
        # l_ref = 33.012 # deg
        ra_ref = 282.25 # deg
        g = 62.6 # deg
        b =  np.arcsin(np.sin(np.deg2rad(dec)) * np.cos(np.deg2rad(g)) - \
                       np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(g)) * np.sin(np.deg2rad(ra) - np.deg2rad(ra_ref)))
        return np.rad2deg(b)


    def field_cut_data(self):
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
        cov_maps = pd.read_csv(self.COV_DIR, delimiter='|', header=0, skiprows=[1], skipfooter=1)
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

    def est_det(self, qpct, opct, skycov, dc=.027):
        if self.DC_override:
            dc = 0.027
        return(skycov * ((qpct * (1-dc) + opct * dc)))

    def make_draw(self, name, pct):
        draw_sims = pd.Series(np.random.random(size=int(1e7)), name=name)
        hits = draw_sims < pct
        hits.name = name
        return hits


    def in_cat_sim(self):
        data = pd.DataFrame(self.df['wd'])
        for survey in self.SURVEYS:
            data = data.merge(self.make_draw(survey, self.wd.loc[data['wd'], survey].reset_index(drop=True)), left_index=True, right_index=True)

        data['in_cat'] = data[self.SURVEYS].sum(axis=1)
        print(data['in_cat'].mean())

    def in_ob_sim(self):
        data = pd.DataFrame(self.df['wd'])
        test_ob = self.make_draw('in_ob', self.wd.loc[self.df['wd'], 'pobs'].reset_index(drop=True))
        print(test_ob.mean())

    def field_observed_sim(self):
        print((~self.df['utobsdate_earliest'].isna()).mean())

    def init_watchdog(self):
        wd = pd.read_csv(self.WATCHDOG_DIR, delim_whitespace=True)

        # turn recurrence times to integers
        wd['trec'] = wd.trecur.str.strip('>').astype(int)

        # lower bound flag
        wd['lb'] = [1 if x == '>' else 0 for x in wd.trecur.str[0]]

        # estimate recurrence time by doubling only those with lower bounds
        wd['trec_est'] = wd['trec'] * (wd['lb'] + 1) - 1

        # find mean outburst duraction
        wd['tout_mean'] = wd['tout'] / wd['Total']

        # change duty cycle from pct to number
        wd['dc'] /= 100

        # calculate pct chance each object is caught by surveys
        for survey in self.SURVEYS:
            wd[f'{survey}'] = [self.est_det(self.PCT_DIS[survey]['Q'], self.PCT_DIS[survey]['O'], self.SKYCOV[survey], dc)for dc in wd['dc']]

        # calculate probability of observing each XRB
        pobs = []
        N = 1505 # days
        rec = 'trec'
        for ii in range(len(wd)):
            trec = wd.loc[ii, rec]
            tout = wd.loc[ii, 'tout_mean']
            Nob = (N + tout) / trec
            pobs.append(1 - poisson.cdf(1, Nob))

        wd['pobs'] = pobs

        return wd

    def create_sim(self):
        pass

if __name__ == '__main__':
    test = XRBsimulator()
    import pdb; pdb.set_trace()
