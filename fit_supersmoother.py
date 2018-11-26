from supersmoother import SuperSmoother
import numpy
from tqdm import trange

lc_list = numpy.load('lc_list.npy')
periods_arr = numpy.load('periods_arr.npy')


grid_len = 200
p_loc = int(grid_len/4)
x_fit = numpy.linspace(0, 1, grid_len)

n_objects = len(lc_list)
X_supersmoother = numpy.zeros([n_objects, grid_len])

for i in trange(n_objects):
    
    lc = lc_list[i]
    mag = lc[:,0]
    dmag = lc[:,2]
    t = lc[:,1]
    period = periods_arr[i]
    phase = (t /period /2) % 1
    
    model = SuperSmoother(alpha=5, period=1)
    model.fit(phase, mag, dmag)
    y_fit = model.predict(x_fit)
    
    grid_order = (x_fit + x_fit[p_loc] - x_fit[numpy.argmax(y_fit)]) %1
    y_fit = y_fit[numpy.argsort(grid_order)]
    
    X_supersmoother[i] = y_fit

numpy.save('X_supersmoother', X_supersmoother)