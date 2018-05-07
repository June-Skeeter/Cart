import pandas as pd 
import numpy as np 
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import metrics
from sklearn import tree
import pydotplus 
from scipy import stats

class Tree:
	def __init__(self,Data,y_var,X_vars,output_path=None):
		self.X_vars = X_vars
		self.Master = Data
		Data = self.Master[np.isfinite(self.Master[y_var])]
		Data = Data.interpolate().bfill()
		Data = Data.interpolate().ffill()
		self.y = Data[y_var].values
		YStandard = StandardScaler()
		self.YScaled = YStandard.fit(self.y.reshape(-1, 1))
		Yscale = self.YScaled.transform(self.y.reshape(-1, 1))
		self.y = np.ndarray.flatten(Yscale)
		self.Ytru = self.YScaled.inverse_transform(self.y.reshape(-1,1))
		X = Data[self.X_vars]
		self.input_shape = len(self.X_vars)
		XStandard = StandardScaler()
		self.XScaled= XStandard.fit(X)
		self.X = self.XScaled.transform(X)

		if output_path is None:
			self.output_path = os.getcwd()
		else:
			self.output_path = output_path

	def Validate_Tree(self,N_Max = 15,samp_size=5,iteration=50,ax=None):
		N = np.linspace(2,N_Max,samp_size,dtype='int32')
		MSE = np.zeros(shape=samp_size)
		STD = np.zeros(shape=samp_size)
		CI = np.zeros(shape=samp_size)
		d = {'N':N,'MSE':MSE,'STD':STD,'CI':CI}
		Runs = pd.DataFrame(data=d)
		logo = LeaveOneGroupOut()
		FI = []
		self.Yfills = []
		for N in Runs['N']:
			MSE = []
			reg = DecisionTreeRegressor(max_leaf_nodes=int(N),random_state=1)
			s = iteration
			fi = []
			y = []
			for i in range(s):
				grp = 10
				group = np.random.permutation(np.arange(self.y.shape[0])) % grp
				for train, test in logo.split(self.X,self.y,groups=group):
					reg.fit(self.X[train],self.y[train])
					y_pred = reg.predict(self.X[test])
					mse = metrics.mean_squared_error(self.y[test],y_pred)
					MSE.append(mse)
					fi.append(reg.feature_importances_)
					y.append(reg.predict(self.X))
			y = np.asanyarray(y).mean(axis=0)
			self.Yfills.append(y)
			fi = np.asanyarray(fi).mean(axis=0)
			FI.append(fi)
			MSE = np.asanyarray(MSE)
			print(MSE.mean(),MSE.std())
			Runs.loc[Runs['N'] == N,['MSE','STD']] = MSE.mean(),MSE.std()
			Runs.loc[Runs['N'] == N,['CI']] =  MSE.std()/(s*10)**.5*stats.t.ppf(1-0.05, s*10-1)
			# dot_data = tree.export_graphviz(reg, out_file=None, feature_names=self.X_vars,  
			# filled=True, rounded=True, special_characters=True) 
			# graph = pydotplus.graph_from_dot_data(dot_data)
			# nodes = graph.get_node_list()
			# graph.write_png(str(N)+'_Test.png')
		self.Features = FI
		ax.bar(Runs['N'],Runs['MSE'],yerr = Runs['CI'])
		print(Runs)


class Forest:
	def __init__(self,Data,y_var,X_vars,output_path=None):
		self.Master = Data
		Data = self.Master[np.isfinite(self.Master[y_var])]
		Data = Data.interpolate().bfill()
		Data = Data.interpolate().ffill()
		self.y = Data[y_var].values
		YStandard = StandardScaler()
		self.YScaled = YStandard.fit(self.y.reshape(-1, 1))
		Yscale = self.YScaled.transform(self.y.reshape(-1, 1))
		self.y = np.ndarray.flatten(Yscale)
		self.Ytru = self.YScaled.inverse_transform(self.y.reshape(-1,1))
		X = Data[X_vars]
		self.input_shape = len(X_vars)
		XStandard = StandardScaler()
		self.XScaled= XStandard.fit(X)
		self.X = self.XScaled.transform(X)

		if output_path is None:
			self.output_path = os.getcwd()
		else:
			self.output_path = output_path
		print(self.output_path)

	def Validate_Tree(self,N_Max = 15,samp_size=5,ax=None):
		N = np.linspace(2,N_Max,samp_size,dtype='int32')
		MSE = np.zeros(shape=samp_size)
		d = {'N':N,'MSE':MSE}
		Runs = pd.DataFrame(data=d)
		logo = LeaveOneGroupOut()
		for N in Runs['N']:
			MSE = []
			reg = RandomForestRegressor(max_leaf_nodes=int(N),random_state=1)
			for i in range(100):
				grp = 10
				group = np.random.permutation(np.arange(self.y.shape[0])) % grp
				for train, test in logo.split(self.X,self.y,groups=group):
					reg.fit(self.X[train],self.y[train])
					y_pred = reg.predict(self.X[test])
					mse = metrics.mean_squared_error(self.y[test],y_pred)
					MSE.append(mse)
			MSE = np.asanyarray(mse)
			print(MSE.mean())
			Runs.loc[Runs['N'] == N,'MSE'] = MSE.mean()
			# dot_data = tree.export_graphviz(reg, out_file=None, feature_names=self.X_vars,  
			# filled=True, rounded=True, special_characters=True) 
			# graph = pydotplus.graph_from_dot_data(dot_data)
			# nodes = graph.get_node_list()
			# graph.write_png(str(N)+'_Test.png')
		ax.bar(Runs['N'],Runs['MSE'])
		print(Runs)