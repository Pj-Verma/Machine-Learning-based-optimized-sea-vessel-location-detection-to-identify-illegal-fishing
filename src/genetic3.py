import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


# def load_dataset(file_address):
#     # Load dataset from file address
#     data = np.load(file_address)
#     # X, y = data['<f8'], data['y']
#     x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(
#         x['speed']) & ~np.isnan(x['course'])]
#
#     return x, y

def threshold_fishing(df):
    df['is_fishing'] = df['is_fishing'].apply(lambda x: 1 if x > 0.5 else 0)
    return df

def fitness_function(params, X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


def genetic_algorithm(X_train, y_train, X_test, y_test, param_grid, pop_size=10, num_generations=20, mutation_rate=0.1):
    population = [{key: np.random.choice(param_grid[key]) for key in param_grid.keys()} for _ in range(pop_size)]

    for generation in range(num_generations):
        fitness = [fitness_function(individual, X_train, y_train, X_test, y_test) for individual in population]

        parents = np.random.choice(population, size=pop_size // 2, replace=False, p=np.array(fitness) / sum(fitness))

        if len(parents) % 2 != 0:
            parents = parents[:-1]

        children = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child = {}
            for key in param_grid.keys():
                if np.random.rand() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            children.append(child)

        for child in children:
            for key in param_grid.keys():
                if np.random.rand() < mutation_rate:
                    child[key] = np.random.choice(param_grid[key])

        children = np.array(children)
        population = np.concatenate((parents, children))

    best_individual = max(population, key=lambda x: fitness_function(x, X_train, y_train, X_test, y_test))
    return best_individual


# Example usage:
def get_all_data(dir = None):
	'''
	it will loop over the files in dir and load them
	input: directory name (srt)
	output: dictionary of dataframes with key = name of file
	'''
	if dir is None:
		dir = os.path.join(os.path.dirname(__file__), 'utils', 'data/labeled')

	datasets = {}
	for filename in os.listdir(dir):
		if filename.endswith('.measures.labels.npz'):
			name = filename[:-len('.measures.labels.npz')]
			#             datasets[name] = dict(zip(['all', 'train', 'cross', 'test'], load_dataset_by_vessel(os.path.join(dir, filename))))
			x = np.load(os.path.join(dir, filename))['x']
			x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]
			datasets[name] = pd.DataFrame(x)
			# print(name)
	return datasets

def get_group(data_dict,gear_type):

	# data_dict = get_all_data('data/labeled')

	group_per_gear = {

	'longliners':
	['alex_crowd_sourced_Drifting_longlines',
	'kristina_longliner_Drifting_longlines',
	'pybossa_project_3_Drifting_longlines'],

	'purse_seines':
	['alex_crowd_sourced_Purse_seines',
	'kristina_ps_Purse_seines',
	'pybossa_project_3_Purse_seines'],

	'trawlers':
	['kristina_trawl_Trawlers',
	'pybossa_project_3_Trawlers'],

	'others':
	['alex_crowd_sourced_Unknown',
	'kristina_longliner_Fixed_gear',
	'kristina_longliner_Unknown',
	'kristina_ps_Unknown',
	'pybossa_project_3_Unknown',
	'pybossa_project_3_Pole_and_line',
	'pybossa_project_3_Trollers'  
	'pybossa_project_3_Fixed_gear'],

	'false_positives':
	['false_positives_Drifting_longlines',
	'false_positives_Fixed_gear',
	'false_positives_Purse_seines',
	'false_positives_Trawlers',
	'false_positives_Unknown']

	}

	df = pd.concat([ data_dict[filename] for filename in group_per_gear[gear_type]])
	df = df.reset_index()
	df = df.drop('index',axis=1)
	return df

# file_address = '../data/labeled/kristina_ps_Purse_seines.measures.labels.npz'  # Replace 'your_dataset.npz' with your dataset file address
# X, y = load_dataset(file_address)
# path='../data/labeled/kristina_ps_Purse_seines.measures.labels.npz'
# path='../data/labeled/alex_crowd_sourced_Drifting_longlines.measures.labels.npz'
# X = np.load(path)['x']
# changed "classification" by "is_fishing"
# X = X[~np.isinf(X['is_fishing']) & ~np.isnan(X['is_fishing']) & ~np.isnan(X['timestamp']) & ~np.isnan(
#     X['speed']) & ~np.isnan(X['course'])]
# df = pd.DataFrame(X)
# y = df['is_fishing'].astype(int).values

# Split the dataset into train and test sets

models_dict = {'model_1': ['course_norm_sin_cos', 'window_1800'],
               'model_2': ['course_norm_sin_cos', 'window_1800', 'window_3600'],
               'model_3': ['course_norm_sin_cos', 'window_1800', 'window_3600', 'window_10800'],
               'model_4': ['course_norm_sin_cos', 'window_1800', 'window_3600', 'window_10800', 'window_21600'],
               'model_5': ['course_norm_sin_cos', 'window_1800', 'window_3600', 'window_10800', 'window_21600',
                           'window_43200'],
               'model_6': ['course_norm_sin_cos', 'window_1800', 'window_3600', 'window_10800', 'window_21600',
                           'window_43200', 'window_86400']}

columns_models_dict = {

    'dist_to_land': [
        'distance_from_port',
        'distance_from_shore',
        'measure_distance_from_port'],

    'course_norm_sin_cos': [
        'measure_course',
        'measure_cos_course',
        'measure_sin_course'],

    # 'measure_pos_1800', took this column out for now.... (feb 2nd) I think it was overfitting to this
    'window_1800': ['measure_coursestddev_1800_log',
                    'measure_daylightavg_1800',
                    'measure_speedstddev_1800',
                    'measure_count_1800',
                    'measure_courseavg_1800',
                    'measure_coursestddev_1800',
                    'measure_speedavg_1800',
                    'measure_speedstddev_1800_log'],

    'window_3600': ['measure_count_3600',
                    'measure_speedstddev_3600',
                    'measure_speedavg_3600',
                    'measure_courseavg_3600',
                    'measure_daylightavg_3600',
                    'measure_coursestddev_3600',
                    'measure_speedstddev_3600_log',
                    'measure_coursestddev_3600_log'],

    'window_10800': ['measure_coursestddev_10800_log',
                     'measure_speedstddev_10800',
                     'measure_speedavg_10800',
                     'measure_daylightavg_10800',
                     'measure_courseavg_10800',
                     'measure_count_10800',
                     'measure_speedstddev_10800_log',
                     'measure_coursestddev_10800'],

    'window_21600': ['measure_coursestddev_21600',
                     'measure_speedavg_21600',
                     'measure_count_21600',
                     'measure_coursestddev_21600_log',
                     'measure_speedstddev_21600_log',
                     'measure_speedstddev_21600',
                     'measure_daylightavg_21600',
                     'measure_courseavg_21600'],

    'window_43200': ['measure_coursestddev_43200',
                     'measure_courseavg_43200',
                     'measure_daylightavg_43200',
                     'measure_coursestddev_43200_log',
                     'measure_speedavg_43200',
                     'measure_count_43200',
                     'measure_speedstddev_43200_log',
                     'measure_speedstddev_43200'],

    'window_86400': ['measure_speedavg_86400',
                     'measure_count_86400',
                     'measure_speedstddev_86400_log',
                     'measure_speedstddev_86400',
                     'measure_coursestddev_86400_log',
                     'measure_coursestddev_86400',
                     'measure_daylightavg_86400',
                     'measure_courseavg_86400'],

}
def is_fishy(x):
        return x['is_fishing'] == 1

def fishy(x):
	return x[is_fishy(x)]

def nonfishy(x):
	return x[~is_fishy(x)]


def _subsample_even(x0, mmsi, n):
	"""Return `n` subsamples from `x0`

	- all samples have given `mmsi`

	- samples are evenly divided between fishing and nonfishing
	"""
	# Create a mask that is true whenever mmsi is one of the mmsi
	# passed in
	mask = np.zeros([len(x0)], dtype=bool)
	for m in mmsi:
		mask |= (x0['mmsi'] == m)
	x = x0[mask]  # this makes is a np array?? nope...

	# Pick half the values from fishy rows and half from nonfishy rows.
	f = fishy(x)
	nf = nonfishy(x)
	if n//2 > len(f) or n//2 > len(nf):
		warnings.warn("insufficient items to sample, returning fewer")
	f_index = np.random.choice(f.index, min(n//2, len(f)), replace=False)
	nf_index = np.random.choice(nf.index, min(n//2, len(nf)), replace=False)

	f = f.loc[f_index]
	nf = nf.loc[nf_index]

	# nf = np.random.choice(nf, min(n//2, len(nf)), replace=False)
	ss = pd.concat([f, nf])  #this was making it a np array! yes
	# np.random.shuffle(ss) # no shuffling
	return ss

def _subsample_proportional(x0, mmsi, n):
	"""Return `n` subsamples from `x0`

	- all samples have given `mmsi`

	- samples are random, so should have ~same be in the same proportions
	  as the x0 for the given mmsi.
	"""
	# Create a mask that is true whenever mmsi is one of the mmsi
	# passed in
	mask = np.zeros([len(x0)], dtype=bool)
	for m in mmsi:
		mask |= (x0['mmsi'] == m)
	x = x0[mask]

	# Pick values randomly
	# Pick values randomly

	# ====DEBUGGER=======
	# import pdb
	# pdb.set_trace()

	if n > len(x):
		warnings.warn("Warning, inufficient items to sample, returning {}".format(len(x)))
		n = len(x)
	x_index = np.random.choice(x.index, n, replace=False)
	# ss = np.random.choice(x, n, replace=False)
	ss = x.loc[x_index]
	# np.random.shuffle(ss) # the shuffeling is giving me trouble
	return ss

def sample_by_vessel(x, size = 20000, even_split=None, seed=4321):

    # def load_dataset_by_vessel(path, size = 20000, even_split=None, seed=4321):
	"""Load a dataset from `path` and return train, valid and test sets

	path - path to the dataset
	size - number of samples to return in total, divided between the
		   three sets as (size//2, size//4, size//4)
	even_split - if True, use 50/50 fishing/nonfishing split for training
				  data, otherwise sample the data randomly.

	The data at path is first randomly divided by divided into
	training (1/2), validation (1/4) and test(1/4) data sets.
	These sets are chosen so that MMSI values are not shared
	across the datasets.

	The validation and test data are sampled randomly to get the
	requisite number of points. The training set is sampled randomly
	if `even_split` is False, otherwise it is chose so that half the
	points are fishing.

	"""
	# Set the seed so that we can reproduce results consistently
	np.random.seed(seed)

	# # Load the dataset and strip out any points that aren't classified
	# # (has'is_fishing == Inf)
	# x = np.load(path)['x']
	# x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]

	if size > len(x):
		print ("Warning, insufficient items to sample, returning all")
		size = len(x)

	# Get the list of MMSI and shuffle them. The compute the cumulative
	# lengths so that we can divide the points ~ evenly. Use search
	# sorted to find the division points
	mmsi = list(set(x['mmsi']))
	if even_split is None:
		even_split = x['is_fishing'].sum() > 1 and x['is_fishing'].sum() < len(x)
	if even_split:
		base_mmsi = mmsi
		# Exclude mmsi that don't have at least one fishing or nonfishing point
		mmsi = []
		for m in base_mmsi:
			subset = x[x['mmsi'] == m]
			fishing_count = subset['is_fishing'].sum()
			if fishing_count == 0 or fishing_count == len(subset):
				continue
			mmsi.append(m)
	np.random.shuffle(mmsi)
	nx = len(x)
	sums = np.cumsum([(x['mmsi'] == m).sum() for m in mmsi])
	n1, n2 = np.searchsorted(sums, [nx//2, 3*nx//4])
	if n2 == n1:
		n2 += 1

	# ====DEBUGGER=======
	# import pdb
	# pdb.set_trace()

	train_subsample = _subsample_even if even_split else _subsample_proportional

# try:
	xtrain = train_subsample(x, mmsi[:n1], size//2)
	# xtrain = _subsample_proportional(x, mmsi[:n1], size//2)
	xcross = _subsample_proportional(x, mmsi[n1:n2], size//4)
	xtest = _subsample_proportional(x, mmsi[n2:], size//4)
	# except Exception, e:
	#     print "==== Broken data in the DataFrame ===="
	#     import pdb, sys
	#     sys.last_traceback = sys.exc_info()[2]
	#     pdb.set_trace()

	return xtrain, xcross, xtest

def keep_columns(df, col_groups):
    from pipeline_noLatLon import columns_models_dict
    # minimal model
    cols_to_keep = ['timestamp', 'mmsi', 'course', 'measure_daylight', 'speed', 'is_fishing']

    if col_groups:

        if col_groups == 'all':
            for key in columns_models_dict:
                cols_to_keep += [col for col in columns_models_dict[key]]
        else:
            for col_g in col_groups:
                cols_to_keep += columns_models_dict[col_g]

    df = df[cols_to_keep]
    # N_cols = len(cols_to_keep)
    return df
def X_y_split(df):
    y = df['is_fishing'].astype(int).values
    df_X = df.drop(['mmsi', 'is_fishing', 'timestamp'], axis=1)
    cols = df_X.columns
    X = df_X.values
    return X, y, cols

gears = ['longliners', 'trawlers', 'purse_seines']
classifiers = ['GEN']
models_by_columns = ['model_1', 'model_2', 'model_3',
                     'model_4', 'model_5', 'model_6']
path = '../data/labeled'
data_dict = get_all_data(path)
for gear in gears:

    df = get_group(data_dict, gear)
    df.reindex()
    df = threshold_fishing(df)
    for model_num in models_by_columns:
            col_groups = models_dict[model_num]
            df_subset = keep_columns(df, col_groups=col_groups)

            df_train, df_cross, df_test = sample_by_vessel(df_subset, size=20000, even_split=None, seed=4321)
            X_train, y_train, cols = X_y_split(df_train)
            X_cross, y_cross, cols = X_y_split(df_cross)
            X_test, y_test, cols = X_y_split(df_test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Call the genetic algorithm
best_params = genetic_algorithm(X_train, y_train, X_test, y_test, param_grid)

# Train final model with best parameters
clf = RandomForestClassifier(**best_params)
clf.fit(X_train, y_train)

# Evaluate the final model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)







