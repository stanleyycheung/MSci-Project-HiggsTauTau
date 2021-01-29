import tensorflow as tf
import datetime
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import kerastuner as kt

class Tuner:
    def __init__(self):
        pass

    
    def runHPTuning(self, config_num, read=True, from_pickle=True, epochs=50, tuner_epochs=50, batch_size=10000, tuner_batch_size=10000, patience=10, tuner_mode=0, addons_config={}):
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tuning on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        best_hps, tuner = self.tuneHP(self.hyperModel, X_train, X_test, y_train, y_test, tuner_epochs=tuner_epochs, tuner_batch_size=tuner_batch_size, tuner_mode=tuner_mode)
        print(tuner.results_summary())
        model = tuner.hypermodel.build(best_hps)
        self.model = model
        # model.fit(X_train, y_train, epochs=epochs, validation_data = (X_test, y_test), verbose=0)
        # verbose is set 0 to prevent logs from spam
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        file = f'{self.write_dir}/best_hp_{self.channel}.txt'
        with open(file, 'a+') as f:
            # print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            best_num_layers = best_hps.get('num_layers')
            # best_batch_norm = best_hps.get('batch_norm')
            # best_dropout = best_hps.get('dropout')
            actual_epochs = len(self.history.history["loss"])
            # message = f'{time_str},{auc},{self.config_num},{best_num_layers},{best_batch_norm},{best_dropout},{tuner_mode}\n'
            message = f'{time_str},{auc},{self.config_num},{best_num_layers},{tuner_mode}\n'
            print(f"Message: {message}")
        #     f.write(message)
        # model.save('./hp_model_1/')


    def runGridSearch(self, config_num, read=True, from_pickle=True, addons_config={}, search_mode=0):
        """
        Runs grid search on NN with given config_num
        search_mode = 0: GridSearch
        search_mode = 1: RandomSearch
        """
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Grid searching on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model = KerasClassifier(self.gridModel, verbose=0)
        if search_mode == 0:
            layers = [2,3,4,5,6]
            batch_norms = [True, False]
            dropouts = [None, 0.2]
            epochs = [100, 200, 500]
            batch_sizes = [8192, 16384, 65536, 131072]
            param_grid = dict(
                layers=layers,
                batch_norm=batch_norms,
                dropout=dropouts,
                epochs=epochs,
                batch_size=batch_sizes
            )
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=2, scoring='roc_auc')
        elif search_mode == 1:
            layers = np.arange(1,11)
            batch_norms = [True, False]
            dropouts = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
            epochs = [50, 100, 200, 500]
            batch_sizes = [2**i for i in range(8, 19)]
            # layers = [2,3]
            # batch_norms = [True, False]
            # dropouts = [None, 0.2]
            # epochs = [200, 500]
            # batch_sizes = [8192, 16384, 65536, 131072]
            param_grid = dict(
                layers=layers,
                batch_norm=batch_norms,
                dropout=dropouts,
                epochs=epochs,
                batch_size=batch_sizes
            )
            # can increase the distributions of params
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=2, verbose=2, scoring='roc_auc', random_state=seed_value, n_iter=20)
        else:
            raise ValueError('Search mode not defined correctly')
        grid_result = grid.fit(X_train, y_train)
        print(grid_result)
        best_num_layers = grid_result.best_params_['layers']
        best_batch_norm = grid_result.best_params_['batch_norm']
        best_dropout = grid_result.best_params_['dropout']
        best_epochs = grid_result.best_params_['epochs']
        best_batchsize = grid_result.best_params_['batch_size']
        self.model = self.gridModel(layers=best_num_layers, batch_norm=best_batch_norm, dropout=best_dropout)
        model = self.train(X_train, X_test, y_train, y_test, epochs=best_epochs, batch_size=best_batchsize, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        file = f'{self.write_dir}/grid_search_{self.channel}.txt'
        with open(file, 'a+') as f:
            print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            message = f'{time_str},{auc},{self.config_num},{best_num_layers},{best_batch_norm},{best_dropout},{best_epochs},{best_batchsize},{search_mode},{grid_result.best_score_},{param_grid}\n'
            print(f"Message: {message}")
            f.write(message)
        model.save(f'./saved_models/grid_search_model_{config_num}_{self.channel}_{search_mode}/')

    def hyperModel(self, hp):
        self.model = tf.keras.models.Sequential()
        num_layers = hp.Int('num_layers', 2, 3)
        self.layers = num_layers
        for i in range(num_layers):
            self.model.add(tf.keras.layers.Dense(units=300, kernel_initializer='normal'))
            # if hp.Boolean('batch_norm', default=False):
                # self.model.add(tf.keras.layers.BatchNormalization())
            # self.model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.2, default=0.0, step=0.2)))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "hyper_model"
        return self.model

    def gridModel(self, layers=2, batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        self.layers = layers
        for i in range(layers):
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


