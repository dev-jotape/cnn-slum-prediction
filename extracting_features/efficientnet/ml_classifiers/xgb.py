import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np

dataset = 'GMAPS_RGB_2024'
x_all = np.load('./results/features_{}.npy'.format(dataset))
y_all = np.load('./results/labels_{}.npy'.format(dataset))
x_all = x_all.reshape(7518, -1)
print(x_all.shape)
# Define the hyperparameter grid
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.1],
    'subsample': [0.5]
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(x_all, [np.argmax(y, axis=None, out=None) for y in y_all])

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Best set of hyperparameters:  {'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.5}
# Best score:  0.8477012287481773