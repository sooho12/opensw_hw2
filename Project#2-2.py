import pandas as pd

def sort_dataset(dataset_df):
	return dataset_df.sort_values('year')

def split_dataset(dataset_df):	
	X = data_df.drop(columns="salary",axis=1)
	y = data_df["salary"]*0.001

	return X[:1718],X[1718:],y[:1718],y[1718:]

def extract_numerical_cols(dataset_df):
	return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
	from sklearn.tree import DecisionTreeRegressor
	dt_rgs = DecisionTreeRegressor()
	dt_rgs.fit(X_train,Y_train)
	return dt_rgs.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
	from sklearn.ensemble import RandomForestRegressor
	rf_rgs = RandomForestRegressor()
	rf_rgs.fit(X_train,Y_train)
	return rf_rgs.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
	from sklearn.svm import SVR
	from sklearn.preprocessing import StandardScaler
	from sklearn.pipeline import make_pipeline
	svm_pipe = make_pipeline(
		StandardScaler(),SVR()
	)
	svm_pipe.fit(X_train,Y_train)
	return svm_pipe.predict(X_test)

def calculate_RMSE(labels, predictions):
	import lightgbm
	lgb_rgs = lightgbm.LGBMRegressor()
	lgb_rgs.fit(X_train,Y_train)
	return lgb_rgs.predict(X_test)

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))