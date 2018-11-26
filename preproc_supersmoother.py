import numpy
from scipy.interpolate import interp1d
from tqdm import trange

X_supersmoother = numpy.load('X_supersmoother.npy')

periods_arr = numpy.load('periods_arr.npy')

n_objects = X_supersmoother.shape[0]
n_features_orig = X_supersmoother.shape[1]

n_features = 1000
grid_all = numpy.linspace(0,2,n_features) #days
n_p = 20

X_supersmoother_same_grid = numpy.zeros([n_objects, n_features])

for i in trange(n_objects):
    period = periods_arr[i]
    
    o_grid = numpy.linspace(0,1,n_features_orig*n_p)*n_p*period*2
    y_dup = numpy.concatenate([X_supersmoother[i]]*n_p)
    f_intr = interp1d(x=o_grid, y=y_dup)
    
    X_supersmoother_same_grid[i] = f_intr(grid_all)
    X_supersmoother_same_grid[i] = X_supersmoother_same_grid[i] - numpy.mean(X_supersmoother_same_grid[i]) 
    #X_supersmoother_same_grid[i] = X_supersmoother_same_grid[i] / numpy.std(X_supersmoother_same_grid[i])
    
numpy.save('X_supersmoother_same_grid_v2', X_supersmoother_same_grid)