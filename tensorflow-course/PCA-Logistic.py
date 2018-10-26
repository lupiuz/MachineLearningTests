import numpy as np 
import pandas as pd 
import tensorflow as ts
import matplotlib.pyplot as plt


from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

mnist = ts.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

train_number = 600
test_number = 600

x_train = x_train[:train_number]
x_test = x_test[:test_number]

y_train = y_train[:train_number]
y_test = y_test[:test_number]

x_train = x_train.flatten().reshape(train_number, 784)
x_train = StandardScaler().fit_transform(x_train)

x_test = x_test.flatten().reshape(test_number, 784)
x_test = StandardScaler().fit_transform(x_test)

#x_train[0]

lr = LogisticRegression(random_state = 0)

pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [10, 20, 50, 100, 300, 500]
}

search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
                      return_train_score=False)
search.fit(x_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(x_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.tight_layout()
plt.show()

# pca = PCA()

# pipe = Pipeline([('pca', pca), ('logistic', lr)])

# param_grid = {'pca__n_components': np.arange(2, x_train.shape[1])}

# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# gs = GridSearchCV(estimator = pipe, 
#                   param_grid = param_grid, 
#                   scoring = scoring, refit = 'AUC', return_train_score = True,
#                   cv = 3)

# gs.fit(x_train, y_train)

# print(gs.best_score_)
# print(gs.best_params_)

# #pipe.fit(x_train, y_train)





# ###############################################################################
# # Plot the PCA spectrum
# pca.fit(x_train)

# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')

# ###############################################################################
# # Prediction

# n_components = [10, 100, 300]
# Cs = np.logspace(-4, 4, 3)

# #Parameters of pipelines can be set using ‘__’ separated parameter names:

# estimator = GridSearchCV(pipe, dict(pca__n_components = n_components, logistic__C = Cs))

# estimator.fit(x_train, y_train)

# plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#             linestyle = ':', label = 'n_components chosen')

# plt.legend(prop = dict(size = 12))
# plt.show()

# #score = pipe.score(x_test)

# #print(score)