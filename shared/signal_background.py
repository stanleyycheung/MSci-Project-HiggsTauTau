import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import config
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval, plotting
import pickle
import utils
import datetime
import json
seed_value = config.seed_value
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# random.seed(seed_value)
# # 3. Set the `numpy` pseudo-random generator at a fixed value
# np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# tf.compat.v1.set_random_seed(seed_value)

# config_tf = tf.compat.v1.ConfigProto()
# config_tf.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config_tf)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

# print("GPU list: ", tf.config.list_physical_devices('GPU'))



class SignalSeperator:
	def __init__(self, mode):
		self.path = '/vols/cms/dw515/Offline/output/SM/masters_signal_vs_background_combined/'
		self.write_path = '/home/hep/shc3117/MSci-Project-HiggsTauTau/shared/signal_background/'
		variables = ['pt_1','pt_2', 'm_vis','svfit_mass','mjj','jpt_1','pt_tt','met','n_jets','pt_vis', 'wt']
		if mode == 0:
			name = 'base'
		elif mode == 1:
			variables += ['iso_1', 'iso_2']
			name = 'iso'
		elif mode == 2:
			name = 'channel'
			variables += ['iso_1', 'iso_2', 'mva_dm_1', 'mva_dm_2', 'tau_decay_mode_2', 'tau_decay_mode_1']
		selectors = ['rand', 'wt_cp_sm', 'deepTauVsJets_medium_1', 'deepTauVsJets_medium_2', 'trg_doubletau', 'os']
		self.name = name
		self.variables = variables
		self.selectors = selectors
		self.mode = mode
		self.scaling = {
			'ggh': 0.0022034,
			'vbfh': 0.0001486,
			'fakes': 1.35,
			'ztt': 3.6292,
		}
		self.cn_value = {
			'higgs': 928.278,
			'ztt': 4.716,
			'fakes': 1.278,
		}

	def run(self):
		self.loadRootFiles()
		df = self.createLabels()
		df_train = df.drop(self.selectors, axis=1)
		X = df_train.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		# print(df_train.drop(['y', 'wt', 'w', 'cn'], axis=1).columns)
		y = df_train.y.astype(int)
		w = df_train.w
		X_train, X_test, y_train, y_test, w_train, w_test, cn_train, cn_test, h_ind_train, h_ind_test = train_test_split(X, y, w, df.cn, df.h_ind, test_size=0.3, random_state=1, stratify=df_train.y)
		eval_set = [(X_train, y_train), (X_test, y_test)]
		xgb_clf = self.XGBModel()
		# xgb_clf.fit(X_train, y_train, sample_weight=w_train, verbose=2, early_stopping_rounds=50, eval_set=eval_set, eval_metric = "mlogloss")
		eval_set = [(X_train, y_train), (X_test, y_test)]
		xgb_clf.fit(X_train, y_train, sample_weight=w_train*cn_train, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w_train*cn_train, w_test*cn_test))
		self.rocCurve(xgb_clf, X_test, y_test, w_test)
		self.confusionMatrix(xgb_clf, X_test, y_test, w_test)
		self.plotScoreDist(xgb_clf, X_test, y_test, w_test)
		sens, count = self.SsqrtB(xgb_clf, X, df_train.w, y, plot=True)
		print(sens, count)

	def runTwoFold(self):
		self.loadRootFiles()
		df = self.createLabels()
		# first half
		df1 = df.iloc[::2]
		df1 = df1[df1.w > 0]
		df_train1 = df1.drop(self.selectors, axis=1)
		X1 = df_train1.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		y1 = df_train1.y.astype(int)
		w1 = df_train1.w
		cn1 = df_train1.cn
		h_ind1 = df_train1.h_ind
		# second half
		df2 = df.iloc[1::2]
		df2 = df2[df2.w > 0]
		df_train2 = df2.drop(self.selectors, axis=1)
		X2 = df_train2.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		y2 = df_train2.y.astype(int)
		w2 = df_train2.w
		cn2 = df_train2.cn
		h_ind2 = df_train2.h_ind
		# training
		xgb_clf1 = self.XGBModel()
		eval_set1 = [(X1, y1), (X2, y2)]
		xgb_clf1.fit(X1, y1, sample_weight=w1*cn1, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set1, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w1*cn1, w2*cn2))
		xgb_clf2 = self.XGBModel()
		eval_set2 = [(X2, y2), (X1, y1)]
		xgb_clf2.fit(X2, y2, sample_weight=w2*cn2, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set2, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w2*cn2, w1*cn1))
		# self.rocCurve(xgb_clf1, X2, y2, w2, subset=1)
		# self.confusionMatrix(xgb_clf1, X2, y2, w2, subset=1)
		# self.plotScoreDist(xgb_clf1, X2, y2, w2, subset=1)
		# self.rocCurve(xgb_clf2, X1, y1, w1, subset=2)
		# self.confusionMatrix(xgb_clf2, X1, y1, w1, subset=2)
		# self.plotScoreDist(xgb_clf2, X1, y1, w1, subset=2)
		# combined eval
		self.rocCurve([xgb_clf1, xgb_clf2], [X1, X2], [y1, y2], [w1, w2])
		self.confusionMatrix([xgb_clf1, xgb_clf2], [X1, X2], [y1, y2], [w1, w2])
		sens, count = self.SsqrtB([xgb_clf1, xgb_clf2], [X1, X2], [w1, w2], [y1, y2], plot=True)
		print(sens, count)

	def runTuning(self):
		space_display = {
            'learning_rate': [np.log(0.01), np.log(0.3)],
            'max_depth': [3, 10, 1],
            'min_child_weight': [1, 13, 2],
            'gamma': [0.0, 1],
            'colsample_bytree': [0.1, 0.6],
            'reg_alpha': [np.log(1e-5), np.log(100)],
        }
		space = {
            'learning_rate': hp.loguniform('learning_rate', *space_display['learning_rate']),
            'max_depth': hp.quniform('max_depth', *space_display['max_depth']),
            'min_child_weight': hp.quniform('min_child_weight', *space_display['min_child_weight']),
            'gamma': hp.uniform('gamma', *space_display['gamma']),
            'colsample_bytree': hp.uniform('colsample_bytree', *space_display['colsample_bytree']),
            'reg_alpha': hp.loguniform('reg_alpha', *space_display['reg_alpha']),
        }
		self.name += '_tuned'
		self.loadRootFiles()
		df = self.createLabels()
		self.df_train = df.drop(self.selectors, axis=1)
		self.X = self.df_train.drop(['y', 'wt', 'w', 'cn'], axis=1).to_numpy()
		self.y = self.df_train.y.astype(int)
		w = self.df_train.w
		self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test, self.cn_train, self.cn_test = train_test_split(self.X, self.y, w, df.cn, test_size=0.3, random_state=1, stratify=self.df_train.y)
		trials = Trials()
		best = fmin(self.hyperOptObjXGB, space, algo=tpe.suggest, trials=trials, max_evals=100)
		best_params = space_eval(space, best)
		file = f"{self.write_path}tuning_results.txt"
		with open(file, 'a+') as f:
			print(f'Writing HPs to {file}')
			time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
			message = f'{time_str},{best_params}\n'
			print(f"Message: {message}")
			f.write(message)
		print(best)
		print(trials.losses())
		print(min(trials.losses()), max(trials.losses()))
		print(best_params)
		print(trials.best_trial)

		pickle.dump(trials, open(f"{self.write_path}trials_{self.name}.p", "wb"))

		# plt.figure()
		# plotting.main_plot_history(trials)
		# plt.savefig(f'{self.write_path}tuning_1_{self.name}.PNG')
		# plt.figure()
		# plotting.main_plot_histogram(trials)
		# plt.savefig(f'{self.write_path}tuning_2_{self.name}.PNG')
		plt.figure(figsize=(16,9))
		# plotting.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		utils.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		plt.savefig(f'{self.write_path}tuning_3_{self.name}.PNG')
		# best_params = {'colsample_bytree': 0.7000000000000001, 'gamma': 0.2, 'learning_rate': 0.05, 'max_depth': 10.0, 'min_child_weight': 2.0, 'n_estimators': 1017.2249974264402, 'reg_alpha': 0.0009900410113592305, 'subsample': 0.6000000000000001}
		self.xgb_clf = self.XGBModel(best_params)
		eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
		self.xgb_clf.fit(self.X_train, self.y_train, sample_weight=self.w_train*self.cn_train, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w_train*self.cn_train, self.w_test*self.cn_test))
		self.rocCurve(self.xgb_clf, self.X_test, self.y_test, self.w_test)
		self.confusionMatrix(self.xgb_clf, self.X_test,self. y_test, self.w_test)
		self.plotScoreDist(self.xgb_clf, self.X_test, self.y_test, self.w_test)
		sens, count = self.SsqrtB(self.xgb_clf, self.X, w, self.y, plot=True)
		# return self.xgb_clf, best_params, space_display
		# print(sens, count)

	def runTuningTwoFold(self):
		space_display = {
            'learning_rate': [np.log(0.01), np.log(0.3)],
            'max_depth': [3, 10, 1],
            'min_child_weight': [1, 13, 2],
            'gamma': [0.0, 1],
            'colsample_bytree': [0.1, 0.6],
            'reg_alpha': [np.log(1e-5), np.log(100)],
        }
		space = {
            'learning_rate': hp.loguniform('learning_rate', *space_display['learning_rate']),
            'max_depth': hp.quniform('max_depth', *space_display['max_depth']),
            'min_child_weight': hp.quniform('min_child_weight', *space_display['min_child_weight']),
            'gamma': hp.uniform('gamma', *space_display['gamma']),
            'colsample_bytree': hp.uniform('colsample_bytree', *space_display['colsample_bytree']),
            'reg_alpha': hp.loguniform('reg_alpha', *space_display['reg_alpha']),
        }
		self.name += '_tuned'
		self.loadRootFiles()
		df = self.createLabels()
				
		# first half
		df1 = df.iloc[::2]
		df1 = df1[df1.w > 0]
		df_train1 = df1.drop(self.selectors, axis=1)
		self.X1 = df_train1.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		self.y1 = df_train1.y.astype(int)
		self.w1 = df_train1.w
		self.cn1 = df_train1.cn
		h_ind1 = df_train1.h_ind
		# second half
		df2 = df.iloc[1::2]
		df2 = df2[df2.w > 0]
		df_train2 = df2.drop(self.selectors, axis=1)
		self.X2 = df_train2.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		self.y2 = df_train2.y.astype(int)
		self.w2 = df_train2.w
		self.cn2 = df_train2.cn
		h_ind2 = df_train2.h_ind
		
		trials = Trials()
		best = fmin(self.hyperOptObjXGBTwoFold, space, algo=tpe.suggest, trials=trials, max_evals=100)
		best_params = space_eval(space, best)
		pickle.dump(trials, open(f"{self.write_path}twofold/trials_{self.name}.p", "wb"))
		plt.figure(figsize=(16,9))
		# plotting.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		utils.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		plt.savefig(f'{self.write_path}twofold/tuning_3_{self.name}.PNG')
		xgb_clf1 = self.XGBModel(best_params)
		eval_set1 = [(self.X1, self.y1), (self.X2, self.y2)]
		xgb_clf1.fit(self.X1, self.y1, sample_weight=self.w1*self.cn1, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set1, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w1*self.cn1, self.w2*self.cn2))
		xgb_clf2 = self.XGBModel(best_params)
		eval_set2 = [(self.X2, self.y2), (self.X1, self.y1)]
		xgb_clf2.fit(self.X2, self.y2, sample_weight=self.w2*self.cn2, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set2, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w2*self.cn2, self.w1*self.cn1))
		self.rocCurve([xgb_clf1, xgb_clf2], [self.X1, self.X2], [self.y1, self.y2], [self.w1, self.w2])
		self.confusionMatrix([xgb_clf1, xgb_clf2], [self.X1, self.X2], [self.y1, self.y2], [self.w1, self.w2])
		sens, count = self.SsqrtB([xgb_clf1, xgb_clf2], [self.X1, self.X2], [self.w1, self.w2], [self.y1, self.y2], plot=True)
		print(sens, count)
		file = f"{self.write_path}twofold/tuning_results.txt"
		with open(file, 'a+') as f:
			print(f'Writing HPs to {file}')
			time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
			message = f'{time_str}-{best_params}-{sens}\n'
			print(f"Message: {message}")
			f.write(message)

	def runClassWeightTuning(self):
		self.loadRootFiles()
		self.vbfh.loc[:, 'w'] = self.vbfh.wt*self.vbfh.wt_cp_sm *self.scaling['vbfh']
		self.ztt.loc[:, 'w'] = self.ztt.wt *self.scaling['ztt']
		self.fakes.loc[:, 'w'] = self.fakes.wt *self.scaling['fakes']
		self.ggh.loc[:, 'w'] = self.ggh.wt*self.ggh.wt_cp_sm*self.scaling['ggh']
		self.vbfh.loc[:, 'h_ind'] = np.ones(self.vbfh.shape[0])
		self.ztt.loc[:, 'h_ind'] = np.zeros(self.ztt.shape[0])
		self.fakes.loc[:, 'h_ind'] = np.zeros(self.fakes.shape[0])
		self.ggh.loc[:, 'h_ind'] = np.full((self.ggh.shape[0],),2.)
		self.higgs = pd.concat([self.vbfh, self.ggh]).reset_index(drop=True)
		self.higgs.loc[:, 'y'] = np.zeros(self.higgs.shape[0])
		self.ztt.loc[:, 'y'] = np.ones(self.ztt.shape[0])
		self.fakes.loc[:, 'y'] = np.full((self.fakes.shape[0],),2.)
		self.higgs.loc[:, 'cn'] = np.full((self.higgs.shape[0],), self.cn_value['higgs'])
		self.ztt.loc[:, 'cn'] = np.full((self.ztt.shape[0],), self.cn_value['ztt'])
		self.fakes.loc[:, 'cn'] = np.full((self.fakes.shape[0],), self.cn_value['fakes'])

		# need to read optimal params
		with open(f"{self.write_path}twofold/tuning_results.txt", 'r') as fh:
			params = [line for line in fh][0]
		self.params_xgb = eval(params.split('-')[1])

		space_display = {
			'higgs_weight': [0, 1],
			'ztt_weight': [0, 1],
		}
		space = {
            'higgs_weight': hp.uniform('higgs_weight', *space_display['higgs_weight']),
            'ztt_weight': hp.uniform('ztt_weight', *space_display['ztt_weight']),
        }
		trials = Trials()
		best = fmin(self.hyperOptClassWeightObj, space, algo=tpe.suggest, trials=trials, max_evals=50)
		best_params = space_eval(space, best)

		pickle.dump(trials, open(f"{self.write_path}class_trials_{self.name}.p", "wb"))
		plt.figure(figsize=(16,9))
		# plotting.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		utils.main_plot_vars(trials, colorize_best=True, arrange_by_loss=True)
		plt.savefig(f'{self.write_path}class_tuning_3_{self.name}.PNG')
		
		result = self.hyperOptClassWeightObj(best_params)
		sens = -result['loss']
		file = f"{self.write_path}class_tuning_results.txt"
		with open(file, 'a+') as f:
			print(f'Writing HPs to {file}')
			time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
			message = f'{time_str}-{best_params}-{sens}-{self.name}\n'
			print(f"Message: {message}")
			f.write(message)

	def loadRootFiles(self):
		vbfh_tree = uproot.open(self.path+'VBFH.root')['ntuple']
		ztt_tree = uproot.open(self.path+'ZTT.root')['ntuple']
		fakes_tree = uproot.open(self.path+'fakes.root')['ntuple']
		ggh_tree = uproot.open(self.path+'ggH.root')['ntuple']
		vbfh = vbfh_tree.pandas.df(self.variables+self.selectors)
		ztt = ztt_tree.pandas.df(self.variables+self.selectors)
		fakes = fakes_tree.pandas.df(self.variables+self.selectors)
		ggh = ggh_tree.pandas.df(self.variables+self.selectors)
		self.vbfh = vbfh[vbfh['deepTauVsJets_medium_1'] & vbfh['deepTauVsJets_medium_2'] & vbfh['trg_doubletau'] & vbfh['os']]
		self.ztt = ztt[ztt['deepTauVsJets_medium_1'] & ztt['deepTauVsJets_medium_2'] & ztt['trg_doubletau'] & ztt['os']]
		self.fakes = fakes[fakes['deepTauVsJets_medium_1'] & fakes['deepTauVsJets_medium_2'] & fakes['trg_doubletau'] & (fakes['os']==0)]
		self.ggh = ggh[ggh['deepTauVsJets_medium_1'] & ggh['deepTauVsJets_medium_2'] & ggh['trg_doubletau'] & ggh['os']]
		
	def createLabels(self):
		self.vbfh.loc[:, 'w'] = self.vbfh.wt*self.vbfh.wt_cp_sm *self.scaling['vbfh']
		self.ztt.loc[:, 'w'] = self.ztt.wt *self.scaling['ztt']
		self.fakes.loc[:, 'w'] = self.fakes.wt *self.scaling['fakes']
		self.ggh.loc[:, 'w'] = self.ggh.wt*self.ggh.wt_cp_sm*self.scaling['ggh']

		self.vbfh.loc[:, 'h_ind'] = np.ones(self.vbfh.shape[0])
		self.ztt.loc[:, 'h_ind'] = np.zeros(self.ztt.shape[0])
		self.fakes.loc[:, 'h_ind'] = np.zeros(self.fakes.shape[0])
		self.ggh.loc[:, 'h_ind'] = np.full((self.ggh.shape[0],),2.)

		self.higgs = pd.concat([self.vbfh, self.ggh]).reset_index(drop=True)
		self.higgs.loc[:, 'y'] = np.zeros(self.higgs.shape[0])
		self.ztt.loc[:, 'y'] = np.ones(self.ztt.shape[0])
		self.fakes.loc[:, 'y'] = np.full((self.fakes.shape[0],),2.)
		self.higgs.loc[:, 'cn'] = np.full((self.higgs.shape[0],), self.cn_value['higgs'])
		self.ztt.loc[:, 'cn'] = np.full((self.ztt.shape[0],), self.cn_value['ztt'])
		self.fakes.loc[:, 'cn'] = np.full((self.fakes.shape[0],), self.cn_value['fakes'])
		df = pd.concat([self.higgs, self.ztt, self.fakes]).reset_index(drop=True)
		df = df[df.w > 0]
		return df

	def plotImportance(self, xgb_clf, df, importance_type='weight'):
		xgboost.plot_importance(xgb_clf, importance_type=importance_type).set_yticklabels(df.drop(['y','wt'], axis=1).columns)
		plt.savefig(f'./signal_background/feature_importance_{importance_type}_{self.name}.PNG')

	def rocCurve(self, xgb_clf, X_test, y_test, w_test, subset=None):
		def plot_roc_curve(fpr, tpr, auc, i):
			ax.plot(fpr, tpr, label=f'{classes[i]} ROC score: {auc:.3f}')
			ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
			plt.legend()
		if isinstance(xgb_clf, list):
			y_pred2 = xgb_clf[0].predict_proba(X_test[1])
			y_pred1 = xgb_clf[1].predict_proba(X_test[0])
			y_pred = np.concatenate([y_pred1, y_pred2])
			w_test = np.append(w_test[0], w_test[1])
			y_test = np.append(y_test[0], y_test[1])
			onehot=label_binarize(y_test, classes=[0,1,2])
		else:
			y_pred = xgb_clf.predict_proba(X_test)
			onehot=label_binarize(y_test,classes=[0,1,2])
		fpr=dict()
		tpr=dict()
		thresh = dict()
		roc_auc=dict()
		for i in range(3):
			fpr[i], tpr[i], thresh[i] = roc_curve(onehot[:,i], y_pred[:,i],sample_weight=w_test)
			roc_auc[i] = auc(fpr[i], tpr[i])
		classes = ['higgs', 'ztt', 'fakes']
		fig, ax = plt.subplots(figsize=(8,6))
		lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
		# ax.set_xlim(lims)
		# ax.set_ylim(lims)
		ax.plot(lims, lims, 'k--')
		ax.grid()
		for i in range(len(fpr)):
			plot_roc_curve(fpr[i],tpr[i],roc_auc[i], i)
		if isinstance(xgb_clf, list):
			plt.savefig(f'{self.write_path}twofold/roc_curves_{self.name}.PNG')
		elif subset is None:
			plt.savefig(f'{self.write_path}roc_curves_{self.name}.PNG')
		else:
			plt.savefig(f'{self.write_path}twofold/roc_curves_{self.name}_{subset}.PNG')

	def confusionMatrix(self, xgb_clf, X_test, y_test, w_test, subset=None):
		def plot_confusion_matrix(y_test, y_pred, classes, figname, w_test=None, normalise_by_col=False, normalise_by_row=False, cmap=plt.cm.Blues):
			if w_test is not None: cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)
			else: cm = confusion_matrix(y_test, y_pred) 
			if normalise_by_col:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print ('Normalised efficiency confusion matrix')
			elif normalise_by_row:
				cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
				print ('Normalised purity confusion matrix')
			else:
				print ('Non-normalised confusion matrix')
			# print(cm)
			plt.figure()
			plt.imshow(cm, interpolation='nearest', cmap=cmap)
			plt.colorbar()
			tick_marks = np.arange(len(classes))
			plt.xticks(tick_marks, classes, rotation=45)
			plt.yticks(tick_marks, classes)

			fmt = '.3f'
			thresh = cm.max() / 2.
			for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, format(cm[i, j], fmt),
						horizontalalignment='center',
						color='w' if cm[i, j] > thresh else 'k')

			plt.tight_layout(pad=1.4)
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			plt.savefig(figname)
			print ('Confusion matrix saved as {}'.format(figname))
			return cm
		
		if isinstance(xgb_clf, list):
			predict2 = xgb_clf[0].predict(X_test[1])
			predict1 = xgb_clf[1].predict(X_test[0])
			predict = np.concatenate([predict1, predict2])
		else:
			predict = xgb_clf.predict(X_test)
		if isinstance(xgb_clf, list):
			figname_purity = f'{self.write_path}twofold/purity_{self.name}.png'
			figname_efficiency = f'{self.write_path}twofold/efficiency_{self.name}.png'
		elif subset is None:
			figname_purity = f'{self.write_path}purity_{self.name}.png'
			figname_efficiency = f'{self.write_path}efficiency_{self.name}.png'
		else:
			figname_purity = f'{self.write_path}twofold/purity_{self.name}_{subset}.png'
			figname_efficiency = f'{self.write_path}twofold/efficiency_{self.name}_{subset}.png'
		if isinstance(xgb_clf, list):
			y_test_comb = np.append(y_test[0], y_test[1])
			w_test_comb = np.append(w_test[0], w_test[1])
			cm_p=plot_confusion_matrix(y_test_comb, predict, [0,1,2],
								normalise_by_row=True, w_test=w_test_comb, figname=figname_purity)
			cm_e=plot_confusion_matrix(y_test_comb, predict, [0,1,2],
								normalise_by_col=True, w_test=w_test_comb, figname=figname_efficiency)
		else:
			cm_p=plot_confusion_matrix(y_test, predict, [0,1,2],
								normalise_by_row=True, w_test=w_test, figname=figname_purity)
			cm_e=plot_confusion_matrix(y_test, predict, [0,1,2],
								normalise_by_col=True, w_test=w_test, figname=figname_efficiency)
		self.write_cm(f'{self.name}', cm_p, cm_e)

	def plotScoreDist(self, xgb_clf, X_test, y_test, w_test, subset=None):
		if isinstance(xgb_clf, list):
			y_pred2 = xgb_clf[0].predict_proba(X_test[1])
			y_pred1 = xgb_clf[1].predict_proba(X_test[0])
			y_pred = np.concatenate([y_pred1, y_pred2])
			w_test = np.append(w_test[0], w_test[1])
			y_test = np.append(y_test[0], y_test[1])
		else:
			y_pred = xgb_clf.predict_proba(X_test)
		labels=['higgs', 'ztt', 'fakes']
		plt.figure()
		for i in range(0, 3):
			plt.hist(y_pred[:, 0][y_test==i], bins=100, weights=w_test[y_test==i], label=labels[i], alpha=0.5, density=True)
			plt.title(labels[0]+' score')
		plt.legend()
		if isinstance(xgb_clf, list):
			plt.savefig(f'{self.write_path}twofold/higgs_score_{self.name}.PNG')
		elif subset is None:
			plt.savefig(f'{self.write_path}higgs_score_{self.name}.PNG')
		else:
			plt.savefig(f'{self.write_path}twofold/higgs_score_{self.name}_{subset}.PNG')
		plt.figure()
		for i in range(0, 3):
			plt.hist(y_pred[:, 1][y_test==i], bins=100, weights=w_test[y_test==i], label=labels[i], alpha=0.5, density=True)
			plt.title(labels[1]+' score')
		plt.legend()
		if isinstance(xgb_clf, list):
			plt.savefig(f'{self.write_path}twofold/ztt_score_{self.name}.PNG')
		elif subset is None:
			plt.savefig(f'{self.write_path}ztt_score_{self.name}.PNG')
		else:
			plt.savefig(f'{self.write_path}twofold/ztt_score_{self.name}_{subset}.PNG')
		plt.figure()
		for i in range(0, 3):
			plt.hist(y_pred[:, 2][y_test==i], bins=100, weights=w_test[y_test==i], label=labels[i], alpha=0.5, density=True)
			plt.title(labels[2]+' score')
		plt.legend()
		if isinstance(xgb_clf, list):
			plt.savefig(f'{self.write_path}twofold/fakes_score_{self.name}.PNG')
		elif subset is None:
			plt.savefig(f'{self.write_path}fakes_score_{self.name}.PNG')
		else:
			plt.savefig(f'{self.write_path}twofold/fakes_score_{self.name}_{subset}.PNG')
		plt.show()

	def SsqrtB(self, xgb_clf, X, weights, y, plot=False, subset=None):
		if isinstance(xgb_clf, list):
			y_pred2 = xgb_clf[0].predict_proba(X[1])
			y_pred1 = xgb_clf[1].predict_proba(X[0])
			y_pred = np.concatenate([y_pred1, y_pred2])
			weights = np.append(weights[0], weights[1])
			y = np.append(y[0], y[1])
			y_classes2 = xgb_clf[0].predict(X[1])
			y_classes1 = xgb_clf[1].predict(X[0])
			y_classes = np.append(y_classes1, y_classes2)
		else:
			y_pred = xgb_clf.predict_proba(X)
			y_classes = xgb_clf.predict(X)
		higgs_events = y_classes==0
		higgs_prob = y_pred[higgs_events][:, 0]
		filtered_mask = (higgs_prob>0.9) & (higgs_prob<1)
		filtered_higgs_prob = higgs_prob[filtered_mask]
		def filterprob(start, end):
			filtered_mask = (higgs_prob>start) & (higgs_prob<end)
			filtered_higgs_prob = higgs_prob[filtered_mask]
			return filtered_mask, filtered_higgs_prob
		if plot:
			fig, ax = plt.subplots(1, 5, sharex='col', sharey='row', figsize=(25,5))
		windows = [[0, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 0.95], [0.95, 1]]
		higgs = []
		ztt = []
		fakes = []
		sensitivity_squared = 0
		for i, w in enumerate(windows):
			filtered_mask, filtered_higgs_prob = filterprob(w[0], w[1])
			S_local = weights[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])][y[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])]==0].sum()
			ztt_local = weights[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])][y[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])]==1].sum()
			fakes_local = weights[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])][y[(y_pred[:, 0]>w[0]) & (y_pred[:, 0]<w[1])]==2].sum()
			B_local = ztt_local + fakes_local
			higgs.append(S_local)
			ztt.append(ztt_local)
			fakes.append(fakes_local)
			sensitivity_squared += (S_local/np.sqrt(max(100, B_local)))**2
			if plot:
				props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
				# ax[i].set_xlim(w[0], w[1])
				textstr= f"S/sqrt(B)={S_local/np.sqrt(B_local):.4f}\nS={S_local:.3f}\nztt={ztt_local:.3f}\nfakes={fakes_local:.3f}"
				ax[i].text(0.05, 0.95, textstr, transform=ax[i].transAxes, fontsize=14, horizontalalignment='left', verticalalignment='top', bbox=props)
				ax[i].hist(filtered_higgs_prob, bins=100)
		if plot:
			plt.subplots_adjust(wspace=0, hspace=0)
			if isinstance(xgb_clf, list):
				plt.savefig(f'{self.write_path}twofold/SsqrtB_plot_{self.name}.PNG')
			elif subset is None:
				plt.savefig(f'{self.write_path}SsqrtB_plot_{self.name}.PNG')
			else:
				plt.savefig(f'{self.write_path}twofold/SsqrtB_plot_{self.name}_{subset}.PNG')
		counts = {
			'higgs': higgs,
			'ztt': ztt,
			'fakes': fakes,
		}
		return np.sqrt(sensitivity_squared), counts

	def XGBModel(self, xgb_params=None):
		if xgb_params is None:
			xgb_params = {
				"objective": "multi:softprob",
				"max_depth": 3,
				"learning_rate": 0.05,
				"silent": 1,
				"n_estimators": 2000,
				"subsample": 0.9,
				"seed": config.seed_value,
				'tree_method':'gpu_hist',
			}
		else:
			xgb_params = {
				"objective": "multi:softprob",
				"seed": config.seed_value,
				'learning_rate': float(xgb_params['learning_rate']),
				'max_depth': int(xgb_params['max_depth']),
				'min_child_weight': int(xgb_params['min_child_weight']),
				'gamma': float(xgb_params['gamma']),
				'colsample_bytree': float(xgb_params['colsample_bytree']),
				'reg_alpha': float(xgb_params['reg_alpha']),
				'subsample': 0.9,
				'n_estimators': 3000,
				'seed': config.seed_value,
				'tree_method':'gpu_hist',
			}
		xgb_clf = xgboost.XGBClassifier(**xgb_params)
		return xgb_clf
	

	def hyperOptObjXGB(self, params):
		params = {
			'learning_rate': float(params['learning_rate']),
			'max_depth': int(params['max_depth']),
			'min_child_weight': int(params['min_child_weight']),
			'gamma': float(params['gamma']),
			'colsample_bytree': float(params['colsample_bytree']),
			'reg_alpha': float(params['reg_alpha']),
		}
		model = self.XGBModel(params)
		eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
		model.fit(self.X_train, self.y_train, sample_weight=self.w_train*self.cn_train, 
						verbose=0, early_stopping_rounds=50, eval_set=eval_set, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w_train*self.cn_train, self.w_test*self.cn_test))
		# sens, _ = self.SsqrtB(model, self.X, self.df_train.w, self.y, plot=False)
		sens, _ = self.SsqrtB(model, self.X_test, self.w_test, self.y_test, plot=False)
		return {'loss': -sens, 'status': STATUS_OK}


	def hyperOptObjXGBTwoFold(self, params):
		params = {
			'learning_rate': float(params['learning_rate']),
			'max_depth': int(params['max_depth']),
			'min_child_weight': int(params['min_child_weight']),
			'gamma': float(params['gamma']),
			'colsample_bytree': float(params['colsample_bytree']),
			'reg_alpha': float(params['reg_alpha']),
		}
		xgb_clf1 = self.XGBModel(params)
		eval_set1 = [(self.X1, self.y1), (self.X2, self.y2)]
		xgb_clf1.fit(self.X1, self.y1, sample_weight=self.w1*self.cn1, 
						verbose=0, early_stopping_rounds=50, eval_set=eval_set1, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w1*self.cn1, self.w2*self.cn2))
		xgb_clf2 = self.XGBModel(params)
		eval_set2 = [(self.X2, self.y2), (self.X1, self.y1)]
		xgb_clf2.fit(self.X2, self.y2, sample_weight=self.w2*self.cn2, 
						verbose=0, early_stopping_rounds=50, eval_set=eval_set2, 
						eval_metric = "mlogloss", sample_weight_eval_set=(self.w2*self.cn2, self.w1*self.cn1))
		sens, _ = self.SsqrtB([xgb_clf1, xgb_clf2], [self.X1, self.X2], [self.w1, self.w2], [self.y1, self.y2], plot=False)
		return {'loss': -sens, 'status': STATUS_OK}


	def hyperOptClassWeightObj(self, params):
		if 1 - params['higgs_weight'] - params['ztt_weight'] < 0:
			return {'loss': 0, 'status': STATUS_OK}
		self.higgs.loc[:, 'cw'] = params['higgs_weight']
		self.ztt.loc[:, 'cw'] = params['ztt_weight']
		self.fakes.loc[:, 'cw'] = 1 - params['higgs_weight'] - params['ztt_weight']
		
		df = pd.concat([self.higgs, self.ztt, self.fakes]).reset_index(drop=True)
		df = df[df.w > 0]
		df1 = df.iloc[::2]
		df1 = df1[df1.w > 0]
		df_train1 = df1.drop(self.selectors, axis=1)
		X1 = df_train1.drop(['y', 'wt', 'w', 'cn', 'h_ind', 'cw'], axis=1).to_numpy()
		y1 = df_train1.y.astype(int)
		w1 = df_train1.w
		cn1 = df_train1.cn
		cw1 = df_train1.cw
		# second half
		df2 = df.iloc[1::2]
		df2 = df2[df2.w > 0]
		df_train2 = df2.drop(self.selectors, axis=1)
		X2 = df_train2.drop(['y', 'wt', 'w', 'cn', 'h_ind', 'cw'], axis=1).to_numpy()
		y2 = df_train2.y.astype(int)
		w2 = df_train2.w
		cn2 = df_train2.cn
		cw2 = df_train2.cw
		# training
		xgb_clf1 = self.XGBModel(self.params_xgb)
		eval_set1 = [(X1, y1), (X2, y2)]
		xgb_clf1.fit(X1, y1, sample_weight=w1*cn1*cw1, 
						verbose=0, early_stopping_rounds=50, eval_set=eval_set1, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w1*cn1*cw1, w2*cn2*cw2))
		xgb_clf2 = self.XGBModel(self.params_xgb)
		eval_set2 = [(X2, y2), (X1, y1)]
		xgb_clf2.fit(X2, y2, sample_weight=w2*cn2*cw2, 
						verbose=0, early_stopping_rounds=50, eval_set=eval_set2, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w2*cn2*cw2, w1*cn1*cw1))
		sens, _ = self.SsqrtB([xgb_clf1, xgb_clf2], [X1, X2], [w1, w2], [y1, y2], plot=False)
		return {'loss': -sens, 'status': STATUS_OK}


	def write_auc(self, config, roc_auc):
		file = self.write_path+'BDT_results.txt'
		with open(file, 'a+') as f:
			aucs = []
			for i in range(0, 4):
				aucs.append(roc_auc[i])
			auc_msg = ','.join(map(str, aucs))
			message = f'{config}; {auc_msg} \n'
			f.write(message)

	def write_cm(self, config, cm_p, cm_e):
		file = self.write_path+'BDT_cm.txt'
		with open(file, 'a+') as f:
			cm_p_msg = ','.join(map(str, cm_p.flatten().tolist()))
			cm_e_msg = ','.join(map(str, cm_e.flatten().tolist()))
			message = f'{config}; {cm_p_msg}; {cm_e_msg} \n'
			f.write(message)


	def runTwoFoldOld(self):
		self.loadRootFiles()
		df = self.createLabels()
		# first half
		df1 = df.iloc[::2]
		df1 = df1[df1.w > 0]
		df_train1 = df1.drop(self.selectors, axis=1)
		X1 = df_train1.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		y1 = df_train1.y.astype(int)
		w1 = df_train1.w
		xgb_clf1 = self.XGBModel()
		X_train1, X_test1, y_train1, y_test1, w_train1, w_test1, cn_train1, cn_test1, h_ind_train1, h_ind_test1 = train_test_split(X1, y1, w1, df1.cn, df1.h_ind, test_size=0.3, random_state=1, stratify=df_train1.y)
		eval_set1 = [(X_train1, y_train1), (X_test1, y_test1)]
		xgb_clf1.fit(X_train1, y_train1, sample_weight=w_train1*cn_train1, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set1, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w_train1*cn_train1, w_test1*cn_test1))
		# second half
		df2 = df.iloc[1::2]
		df2 = df2[df2.w > 0]
		df_train2 = df2.drop(self.selectors, axis=1)
		X2 = df_train2.drop(['y', 'wt', 'w', 'cn', 'h_ind'], axis=1).to_numpy()
		y2 = df_train2.y.astype(int)
		w2 = df_train2.w
		xgb_clf2 = self.XGBModel()
		X_train2, X_test2, y_train2, y_test2, w_train2, w_test2, cn_train2, cn_test2, h_ind_train2, h_ind_test2 = train_test_split(X2, y2, w2, df2.cn, df2.h_ind, test_size=0.3, random_state=1, stratify=df_train2.y)
		eval_set2 = [(X_train2, y_train2), (X_test2, y_test2)]
		xgb_clf2.fit(X_train2, y_train2, sample_weight=w_train2*cn_train2, 
						verbose=2, early_stopping_rounds=50, eval_set=eval_set2, 
						eval_metric = "mlogloss", sample_weight_eval_set=(w_train2*cn_train2, w_test2*cn_test2))

if __name__ == '__main__':
	SB = SignalSeperator(mode=2)
	# SB.run()
	# SB.runTuning()

	# SB.runTwoFold()
	SB.runTuningTwoFold()

	# SB.runClassWeightTuning()