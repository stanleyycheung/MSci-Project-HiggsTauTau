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
		if mode == 1:
			variables += ['iso_1', 'iso_2']
			name = 'iso'
		elif mode == 2:
			variables += ['iso_1', 'iso_2', 'aco_angle_1']
			name = 'aco_angle'
		elif mode == 3:
			variables += ['iso_1', 'iso_2', 'aco_angle_1', 'y_1_1', 'pi_E_1', 'pi0_E_1']
			name = 'all'
		selectors = ['rand', 'wt_cp_sm', 'deepTauVsJets_medium_1', 'deepTauVsJets_medium_2', 'trg_doubletau', 'os']
		self.name = name
		self.variables = variables
		self.selectors = selectors
		self.mode = mode


	def run(self):
		self.loadRootFiles()
		df = self.createLabels()
		df_train = df.drop(self.selectors, axis=1)
		X = df_train.drop(['y', 'wt'], axis=1).to_numpy()
		y = df_train.y.astype(int)
		w = df_train.wt
		X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=1, stratify=df_train.y)
		eval_set = [(X_train, y_train), (X_test, y_test)]
		xgb_clf = self.XGBModel()
		xgb_clf.fit(X_train, y_train, sample_weight=w_train, verbose=2, early_stopping_rounds=50, eval_set=eval_set, eval_metric = "mlogloss")
		self.rocCurve(xgb_clf, X_test, y_test)
		self.confusionMatrix(xgb_clf, X_test, y_test, w_test)

	def loadRootFiles(self):
		vbfh_tree = uproot.open(self.path+'VBFH.root')['ntuple']
		ztt_tree = uproot.open(self.path+'ZTT.root')['ntuple']
		fakes_tree = uproot.open(self.path+'fakes.root')['ntuple']
		ggh_tree = uproot.open(self.path+'ggH.root')['ntuple']
		vbfh = vbfh_tree.pandas.df(self.variables+self.selectors)
		ztt = ztt_tree.pandas.df(self.variables+self.selectors)
		fakes = fakes_tree.pandas.df(self.variables+self.selectors)
		ggh = ggh_tree.pandas.df(self.variables+self.selectors)
		vbfh = vbfh[vbfh['deepTauVsJets_medium_1'] & vbfh['deepTauVsJets_medium_2'] & vbfh['trg_doubletau'] & vbfh['os']]
		self.ztt = ztt[ztt['deepTauVsJets_medium_1'] & ztt['deepTauVsJets_medium_2'] & ztt['trg_doubletau'] & ztt['os']]
		self.fakes = fakes[fakes['deepTauVsJets_medium_1'] & fakes['deepTauVsJets_medium_2'] & fakes['trg_doubletau'] & (fakes['os']==0)]
		ggh = ggh[ggh['deepTauVsJets_medium_1'] & ggh['deepTauVsJets_medium_2'] & ggh['trg_doubletau'] & ggh['os']]
		self.vbfh_sm = vbfh[(vbfh["rand"] < vbfh["wt_cp_sm"]/2)]
		self.ggh_sm = ggh[(ggh["rand"] < ggh["wt_cp_sm"]/2)]

	def createLabels(self):
		self.vbfh_sm.loc[:, 'y'] = np.zeros(self.vbfh_sm.shape[0])
		self.ztt.loc[:, 'y'] = np.ones(self.ztt.shape[0])
		self.fakes.loc[:, 'y'] = np.full((self.fakes.shape[0],),2.)
		self.ggh_sm.loc[:, 'y'] = np.full((self.ggh_sm.shape[0],),3.)
		self.vbfh_sm = self.vbfh_sm[(self.vbfh_sm != -9999).all(1)]
		self.ztt = self.ztt[(self.ztt != -9999).all(1)]
		self.fakes = self.fakes[(self.fakes != -9999).all(1)]
		self.ggh_sm = self.ggh_sm[(self.ggh_sm != -9999).all(1)]
		df = pd.concat([self.vbfh_sm, self.ztt, self.fakes, self.ggh_sm]).reset_index(drop=True)
		df = df[(df >= 0).all(1)]
		return df

	def plotImportance(self, xgb_clf, df, importance_type='weight'):
		xgboost.plot_importance(xgb_clf, importance_type=importance_type).set_yticklabels(df.drop(['y','wt'], axis=1).columns)
		plt.savefig(f'./signal_background/feature_importance_{importance_type}_{self.name}.PNG')

	def rocCurve(self, xgb_clf, X_test, y_test):
		y_pred = xgb_clf.predict_proba(X_test)
		onehot=label_binarize(y_test, classes=[0,1,2,3])
		fpr=dict()
		tpr=dict()
		thresh = dict()
		roc_auc=dict()
		for i in range(4):
			fpr[i], tpr[i], thresh[i] = roc_curve(onehot[:,i], y_pred[:,i])#,None,w_test)
			roc_auc[i] = auc(fpr[i], tpr[i])#,reorder=True)
			
		classes = ['vbfH', 'ztt', 'fakes', 'ggH']
		self.write_auc(f'{self.name}', roc_auc)
		fig, ax = plt.subplots(figsize=(8,6))
		def plot_roc_curve(fpr, tpr, auc, i):
			ax.plot(fpr, tpr, label=f'{classes[i]} ROC score: {auc:.3f}')
			ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
			plt.legend()
		lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
		ax.set_xlim(lims)
		ax.set_ylim(lims)
		ax.plot(lims, lims, 'k--')
		ax.grid()
		for i in range(len(fpr)):
			plot_roc_curve(fpr[i],tpr[i],roc_auc[i], i)
		plt.savefig(f'{self.write_path}roc_curves_{self.name}.PNG')

	def confusionMatrix(self, xgb_clf, X_test, y_test, w_test):
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
			print(cm)
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
		predict=xgb_clf.predict(X_test)
		cm_p=plot_confusion_matrix(y_test, predict, [0,1,2,3],
                    normalise_by_row=True, w_test=w_test, figname=f'{self.write_path}purity_{self.name}.png')
		cm_e=plot_confusion_matrix(y_test, predict, [0,1,2,3],
							normalise_by_col=True, w_test=w_test, figname=f'{self.write_path}efficiency_{self.name}.png')
		self.write_cm(f'{self.name}', cm_p, cm_e)

	def XGBModel(self):
		xgb_params = {
			"objective": "multi:softprob",
			"max_depth": 3,
			"learning_rate": 0.05,
			"silent": 1,
			"n_estimators": 1000,
			"subsample": 0.9,
			"seed": 123451,
		}
		xgb_clf = xgboost.XGBClassifier(**xgb_params)
		return xgb_clf
	

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


if __name__ == '__main__':
	SB = SignalSeperator(mode=3)
	SB.run()