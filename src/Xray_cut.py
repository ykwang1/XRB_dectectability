import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

observatories = ['ROSAT', 'Swift', 'Chandra', 'XMM']
surveys = ['ROSAT', 'Swift', 'Chandra', 'XMM', 'XMMSL']
QLUM = 10 ** 31 * u.erg
OLUM = 10 ** 38 * u.erg
SIM_POSITIONS = '../data/sample_xrb_positions_10M.csv.gz'
EXPOSURES = '../data/exposures.csv'

# Xray survey extinction
WebPIMMs_output = {
    'ROSAT' :
        pd.DataFrame([(1e-15, 0, 2.276e-5, 1.173e-4),
        (1e-15, 3e19, 1.965e-5, 9.185e-5),
        (1e-15, 3e20, 1.184e-5, 3.659e-5),
        (1e-15, 3e21, 5.606e-6, 1.486e-5),
        (1e-15, 1e22, 2.190e-6, 5.958e-6),
        (1e-15, 2e22, 8.617e-7, 2.494e-6),
        (1e-15, 3e22, 4.139e-7, 1.263e-6),
        (1e-15, 6e22, 7.425e-8, 2.528e-7),
        (1e-15, 1e23, 1.200e-8, 4.433e-8),
        ], columns=['flux', 'nH', 'HRI', 'PSPC'])
        ,

    'Chandra' :
        pd.DataFrame([(1e-15, 0, 3.066e-5, 4.083e-5),
        (1e-15, 3e19, 3.062e-5, 4.055e-5),
        (1e-15, 3e20, 3.020e-5, 3.887e-5),
        (1e-15, 3e21, 2.672e-5, 3.322e-5),
        (1e-15, 1e22, 2.084e-5, 2.520e-5),
        (1e-15, 2e22, 1.614e-5, 1.907e-5),
        (1e-15, 3e22, 1.336e-5, 1.552e-5),
        (1e-15, 6e22, 9.173e-6, 1.024e-5),
        (1e-15, 1e23, 3.696e-6, 7.209e-6)
        ], columns=['flux', 'nH', 'ACIS-I', 'ACIS-S'])
        ,

    'XMM':
        pd.DataFrame([(1e-15, 0, 5.076e-4, 3.017e-4),
        (1e-15, 3e19, 4.838e-4, 2.956e-4),
        (1e-15, 3e20, 3.586e-4, 2.564e-4),
        (1e-15, 1e21, 2.527e-4, 2.042e-4),
        (1e-15, 3e21, 1.617e-4, 1.414e-4),
        (1e-15, 1e22, 8.717e-5, 8.088e-5),
        (1e-15, 2e22, 5.900e-5, 5.612e-5),
        (1e-15, 3e22, 4.635e-5, 4.475e-5),
        (1e-15, 6e22, 3.010e-5, 3.112e-5),
        (1e-15, 1e23, 2.167e-5, 2.190e-5)
        ], columns=['flux', 'nH', 'PN-THIN', 'PN-THICK'])
        ,

    'Swift':
        pd.DataFrame([(1e-15, 0, 2.222e-5, 2.276e-5),
        (1e-15, 3e19, 2.199e-5, 2.696e-5),
        (1e-15, 3e20, 2.016e-5, 2.464e-5),
        (1e-15, 3e21, 1.26e-5, 1.481e-5),
        (1e-15, 1e22, 7.772e-6, 8.93e-6),
        (1e-15, 2e22, 5.484e-6, 6.28e-6),
        (1e-15, 3e22, 4.370e-6, 5.009e-6),
        (1e-15, 6e22, 2.860e-6, 3.301e-6),
        (1e-15, 1e23, 2.033e-6, 2.366e-6)
        ], columns=['flux', 'nH', 'XRT PC', 'XRT PD'])
}

class Interpolate():
    """Interpolates webPIMMs counts to get cps as a function of flux and nH"""
    def __init__(self, WP_output, instr):
        self.data = WP_output
        self.instr = instr

        # use 1-d interpolation to fill in the extinction at each column density
        self.cps_interp = interp1d(self.data['nH'], self.data[instr], kind='quadratic')

        # the flux at which the counts in WP_output are corresponding to
        self.data_flux = 1e-15

        # check that the flux matches the fluxes in the WP_output
        assert self.data_flux == self.data['flux'].values[0]

    def check_interp(self):
        '''Plot the interpolation from self.cps_interp against the WebPIMMS output'''
        fig, ax = plt.subplots()
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log")
        self.data.plot(x='nH', y=self.instr, ax=ax)
        xs = np.logspace(19, 23, 100)
        ax.plot(xs, self.cps_interp(xs))
        plt.ylabel('Counts/s')
        plt.show()

    def get_cps(self, nH, flux):
        '''Returns the expected counts/s accounting for extinction due to nH'''
        return (self.cps_interp(nH) * flux / self.data_flux).astype(float) # multiply

    def plot_cps_distr(self, nH, flux, state='Q'):
        pass

def find_ev_bootstrap(Qcps, exp, snr_thres = 25):
    '''find the number of objects with counts greater than the snr_thres'''
    exp_samp = np.random.choice(exp, size=len(Qcps))
    snr = (Qcps * exp_samp) / np.sqrt(Qcps * exp_samp)
    # plot_loghist(snr)
    return sum(snr > snr_thres) # / len(snr)

def main(threshold=25):
    observatories = ['ROSAT', 'Swift', 'Chandra', 'XMM']
    surveys = ['ROSAT', 'Swift', 'Chandra', 'XMM', 'XMMSL']

    # initialize interpolators
    instruments = {'ROSAT': 'PSPC', 'Swift': 'XRT PC', 'Chandra': 'ACIS-S', 'XMM': 'PN-THIN'}
    interpolators = {obs:Interpolate(WebPIMMs_output[obs], instruments[obs]) for obs in observatories}

    # read in simulated positions and find column density from extinction
    df = pd.read_csv(SIM_POSITIONS, header=None, names=['ra', 'dec', 'd_kpc', 'E(B-V)'])

    # read in survey exposure times
    exp = pd.read_csv(EXPOSURES)
    exposures = {survey: exp[survey].dropna().values for survey in surveys}

    # estimate column density
    N = len(df)
    R_V = 3.1  #  A_V = R_V * E(B-V)
    df['nH'] = 2.21 * 10 ** 21 * R_V * df['E(B-V)'] # 2.21 * 10 ** 21  * A_V

    # Calculate fluxes from luminosity and distance
    df['Oflux'] = OLUM * u.erg / (df['d_kpc'].values ** 2 * u.kpc ** 2).to(u.cm**2).value
    df['Qflux'] = QLUM * u.erg / (df['d_kpc'].values ** 2 * u.kpc ** 2).to(u.cm**2).value

    # get counts/s expected using simulated data and extinctions
    for obs in observatories:
        for state in 'QO':
            df[f'{state}cps_{obs}'] = interpolators[obs].get_cps(df['nH'], df[f'{state}flux'])

    # get percent discoverable
    pct_dis = {survey:{} for survey in surveys}
    for survey in surveys:
        # set obs to the matching "survey"
        obs = survey
        if survey == 'XMMSL':
            obs = 'XMM'

        # Calculate percentage of XRBs would  be visible in each state using 25 counts as a threshold
        for state in 'QO':
            pct_dis[survey][state] = round(find_ev_bootstrap(df[f'{state}cps_{obs}'], exposures[survey]) / N, 6) # * 1e2
        #
        print(f'{survey}\t{pct_dis[survey]["Q"]}\t{pct_dis[survey]["O"]}')
    print(pct_dis)
    return pct_dis

if __name__ == '__main__':
    main()
