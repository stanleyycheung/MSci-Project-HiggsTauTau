import os
import hyperopt
import tensorflow as tf
import datetime
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import kerastuner as kt
import config
import random
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
seed_value = config.seed_value
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)
from sklearn.metrics import roc_curve, roc_auc_score

class Tuner:
    def __init__(self, mode='random_sk'):
        """
        mode:
        'random_sk': RandomizedSearchCV (default)
        'grid_search_cv': GridSearchCV
        'hyperband': HyperBand
        'bayesian': BayesianOptimization
        'random_kt': RandomSearch
        """
        
        # if mode not in {'random_sk', 'grid_search_cv', 'hyperband', 'bayesian', 'random_kt'}:
        #     raise ValueError('Please choose valid tuner mode')
        self.mode = mode
        # smaller parameter space - always runs in GridSearchCV
        self.param_grid_1 = dict(
            layers=[2, 3, 4, 5, 6],
            batch_norm=[True, False],
            dropout=[None, 0.2],
            epochs=[100, 200, 500],
            batch_size= [8192, 16384, 65536, 131072]
        )
        # larger parameter space - always runs in GridSearchCV
        self.param_grid_2 = dict(
            layers=np.arange(1, 11).tolist(),
            batch_norm=[True, False],
            dropout=[None, 0.1, 0.2, 0.3, 0.4, 0.5],
            epochs=[50, 100, 200, 500],
            batch_size=[2**i for i in range(8, 19)]
        )

    def tune(self, X_train, y_train, X_test, y_test):
        print(f'Tuning using {self.mode}')
        if self.mode in {'random_sk', 'grid_search_cv'}:
            model = KerasClassifier(self.gridModel, verbose=0)
            if self.mode == 'grid_search_cv':
                param_grid = self.param_grid_1
                param_grid = dict(
                    layers=np.arange(2, 7, 2).tolist(),
                    # layers=np.arange(1, 11).tolist(),
                    batch_norm=[True, False],
                    dropout=[None, 0.2],
                    epochs=[50],
                    batch_size=[8192, 16384],
                    # batch_size= [2**i for i in range(10, 19)]
                )
                grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=2, scoring='roc_auc', n_jobs=2)
            else:
                # param_grid = self.param_grid_2
                param_grid = dict(
                    layers=np.arange(1, 7).tolist(),
                    # layers=np.arange(1, 11).tolist(),
                    batch_norm=[True, False],
                    dropout=[None, 0.2, 0.4],
                    epochs=[50, 100],
                    batch_size=[2**i for i in range(8, 19, 2)],
                    # batch_size= [2**i for i in range(10, 19)]
                )
                # layers = np.array([2,3]).tolist()
                # batch_norms = [True, False]
                # dropouts = [None, 0.2]
                # epochs = [1]
                # batch_sizes = [10000]
                # param_grid = dict(
                #     layers=layers,
                #     batch_norm=batch_norms,
                #     dropout=dropouts,
                #     epochs=epochs,
                #     batch_size=batch_sizes
                # )

                grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, verbose=10, scoring='roc_auc', random_state=seed_value, n_iter=20, cv=2, n_jobs=2)
            grid_result = grid.fit(X_train, y_train)
            model_grid = self.gridModel(layers=grid_result.best_params_['layers'], batch_norm=grid_result.best_params_['batch_norm'], dropout=grid_result.best_params_['dropout'])
            return model_grid, grid_result, param_grid
        elif self.mode in {'hyperband', 'bayesian', 'random_kt'}:
            if self.mode == 'hyperband':
                tuner = kt.Hyperband(self.hyperModel,
                                    objective=kt.Objective("auc", direction="max"),  # ['loss', 'auc', 'accuracy', 'val_loss', 'val_auc', 'val_accuracy']
                                    max_epochs=200,
                                    hyperband_iterations=3,
                                    factor=3,
                                    seed=seed_value,
                                    directory='tuning',
                                    project_name='model_hyperband',
                                    overwrite=True)
            elif self.mode == 'bayesian':
                tuner = kt.BayesianOptimization(self.hyperModel,
                                                objective='val_loss',
                                                max_trials=100,
                                                seed=seed_value,
                                                directory='tuning',
                                                project_name='model_bayesian',
                                                overwrite=True)
            elif self.mode=='random_kt':
                tuner = kt.RandomSearch(self.hyperModel,
                                        objective='val_loss',
                                        max_trials=20,
                                        seed=seed_value,
                                        directory='tuning',
                                        project_name='model_random',
                                        overwrite=True)
            else:
                raise ValueError('Invalid tuner mode')
            tuner.search(X_train, y_train, epochs=50, batch_size=10000, validation_data=(X_test, y_test), verbose=1)
            print(tuner.search_space_summary())
            # tuner.search(X_train, y_train, epochs=tuner_epochs, validation_data=(X_test, y_test), verbose=1)
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(tuner.results_summary())
            model = tuner.hypermodel.build(best_hps)
            return model, best_hps, None
        elif self.mode in {'hyperopt'}:
            # in hyperopt territory
            # space = {
            #     'num_layers': hp.choice('num_layers', np.arange(1, 11)),
            #     'batch_norm': hp.choice('batch_norm', [True, False]),
            #     'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            #     'epochs': hp.choice('epochs', [50, 100, 200, 500]),
            #     'batch_size': hp.choice('batch_size', [2**i for i in range(8, 19)]),
            # }
            space_display = {
                'num_layers': [2, 6, 1],
                'batch_norm': [True, False],
                'dropout': [0, 0.4, 0.1],
                'epochs': [20, 50, 10],
                'batch_size': [8192, 16384, 65536, 131072],
            }
            space = {
                'num_layers': hp.quniform('num_layers', *space_display['num_layers']),
                'batch_norm': hp.choice('batch_norm', space_display['batch_norm']),
                'dropout': hp.quniform('dropout', *space_display['dropout']),
                'epochs': hp.quniform('epochs', *space_display['epochs']),
                'batch_size': hp.choice('batch_size', space_display['batch_size'])
            }
            self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
            trials = Trials()
            best = fmin(self.hyperOptObj, space, algo=tpe.suggest, trials=trials, max_evals=10)
            best_params = hyperopt.space_eval(space, best)
            model, _ = self.hyperOptModel(best_params)
            # print(best_params)
            # print(trials.best_trial)
            return model, best_params, space_display
        else:
            raise ValueError('Tuning mode not valid')


    def hyperModel(self, hp):
        # TODO: Change model
        self.model = tf.keras.models.Sequential()
        num_layers = hp.Int('num_layers', 2, 6)
        self.layers = num_layers
        for i in range(num_layers):
            self.model.add(tf.keras.layers.Dense(units=300, kernel_initializer='normal'))
            if hp.Boolean('batch_norm', default=False):
                self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.2, default=0.0, step=0.2)))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "hyper_model"
        return self.model

    def gridModel(self, layers=2, batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        self.layers = layers
        for _ in range(layers):
            self.model.add(tf.keras.layers.Dense(300, kernel_initializer='normal'))
            if batch_norm:
                self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            if dropout is not None:
                self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "grid_model"
        return self.model

    def hyperOptModel(self, params):
        params = {
                'num_layers': int(params['num_layers']),
                'batch_norm': bool(params['batch_norm']),
                'dropout': int(params['dropout']),
                'epochs': int(params['epochs']),
                'batch_size': int(params['batch_size']),
            }
        model = tf.keras.models.Sequential()
        for _ in range(params['num_layers']):
            model.add(tf.keras.layers.Dense(300, kernel_initializer='normal'))
            if params['batch_norm']:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            if not params['dropout']:
                model.add(tf.keras.layers.Dropout(params['dropout']))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=10)
        return model, early_stop

    def hyperOptObj(self, params):
        params = {
                'num_layers': int(params['num_layers']),
                'batch_norm': bool(params['batch_norm']),
                'dropout': int(params['dropout']),
                'epochs': int(params['epochs']),
                'batch_size': int(params['batch_size']),
            }
        model, early_stop = self.hyperOptModel(params)
        model.fit(self.X_train, self.y_train, epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[early_stop], verbose=0)
        y_proba = model.predict(self.X_test)  # outputs two probabilties
        auc = roc_auc_score(self.y_test, y_proba)
        return {'loss': -auc, 'status': STATUS_OK}

if __name__ == "__main__":
    pass