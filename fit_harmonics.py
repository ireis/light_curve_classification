from scipy.optimize import curve_fit
import numpy
global w
from tqdm import trange
print('########################################')
print('########################################')
print('########################################\n')
print('Loading light curves....\n')
lc_list = numpy.load('lc_list.npy')
periods = numpy.load('periods_arr.npy')
print('Done loading.\n')
print('########################################')
print('########################################')
print('########################################\n')
n_objects = len(lc_list)


def func(x, A0, A1, B1, A2, B2, A3, B3, A4, B4):
    return A0 + A1*numpy.sin(w*x) + B1*numpy.cos(w*x) + A2*numpy.sin(2*w*x) + B2*numpy.cos(2*w*x) + A3*numpy.sin(3*w*x) + B3*numpy.cos(3*w*x) + A4*numpy.sin(4*w*x) + B4*numpy.cos(4*w*x) 

params_list = []
corr_list = []
print('Fitting {} light curves:'.format(n_objects))
for idx in trange(n_objects):
    lc = lc_list[idx]
    mag = lc[:,0]
    dmag = lc[:,2]
    t = lc[:,1]
    w = 2*numpy.pi/periods[idx]
    params, corr = curve_fit(func, t, mag, sigma=dmag, absolute_sigma=False)
    params_list += [params]
    corr_list += [corr]

print('Done fitting.\n')

path_params = 'fit_params_4h.npy'
path_corr = 'fit_corrs_4h.npy'
numpy.save(path_params, params_list)
numpy.save(path_corr, corr_list)
print('Saved fit parameters and correlations matrices to {}, {}'.format(path_params, path_corr))

print('Done!')