
import gatspy
from gatspy.periodic import LombScargleFast
from tqdm import trange
import numpy

lc_list = numpy.load('lc_list.npy')


def get_period(t, mag, dmag, n_periods = 1):
    model = LombScargleFast().fit(t, mag, dmag)
    periods, power = model.periodogram_auto(nyquist_factor=200)
    model.optimizer.period_range=(periods.min(), numpy.min([periods.max(), 2]))
    model.optimizer.quiet = True
    best_period, best_period_scores = model.optimizer.find_best_periods(model, n_periods=n_periods, return_scores=True)
    return best_period, best_period_scores

def get_period_single(lc):
    mag = lc[:,0]
    dmag = lc[:,2]
    t = lc[:,1]
    n_periods = 5
    best_period, best_period_scores = get_period(t, mag, dmag, n_periods)
    return best_period[0]

    
    
from joblib import Parallel, delayed
periods = Parallel(n_jobs=-1, verbose=10)(delayed(get_period_single)
                                                    (lc) for lc in lc_list)
    

periods_arr = numpy.array(periods)
numpy.save('periods_arr', periods_arr)