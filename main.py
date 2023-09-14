import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import pickle

import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, Sequential, models

DATASET_DIR = 'Dataset/'

def load_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
	df = pd.read_csv(DATASET_DIR + 'train.csv').set_index('sample_id')

	X = df.iloc[:, 3:].copy()
	y = df['land_cover']

	return train_test_split(X, y, test_size=0.2, stratify=y)

def build_model(X_train: np.array) -> tf.keras.Model:
	model = Sequential([
		layers.Input(shape=(X_train.shape[1], ), dtype='float64'),
		layers.Dense(1440, activation='relu'),
		layers.Dense(720, activation='sigmoid'),
		layers.Dense(360, activation='relu'),
		layers.Dense(72, activation='softmax')
	])

	model.compile(optimizer=optimizers.Adam(),
				loss=losses.SparseCategoricalCrossentropy(),
				metrics=['accuracy']
	)

	return model

def train_model(model: tf.keras.Model, X_train: np.array, X_val: np.array, y_train: np.array, y_val: np.array, save:bool=False) -> float:
	base_f1 = -1
	best_model = 0
	y_train = y_train.astype('float64').to_numpy()

	for batch in [72,720,15000]:
		for i in range(5):
			model.fit(
				X_train,
				y_train,
				epochs=1,
				batch_size=batch,
				validation_data=(X_val, y_val)
			)

			y_pred = model.predict(X_val, verbose=0)
			curr_f1 = f1_score(y_val, np.argmax(y_pred, axis=1), average='weighted')
			print(f'F1 score for {batch} / {i}: ', curr_f1)

			if base_f1 < curr_f1:
				best_model = models.clone_model(model)
				best_model.build((None, 1)) 
				best_model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy())
				best_model.set_weights(model.get_weights())
				base_f1 = curr_f1

	print('Model Finished')
	print('Best Model score:', base_f1)

	if save:
		with open(f'TF_model_val_score-{base_f1}.pkl', 'wb') as f:
			pickle.dump(model, f)

	return base_f1

def write_submission_file(model: tf.keras.Model, transformers:list, file_append:str) -> None:
	test  = pd.read_csv(DATASET_DIR + "test.csv", index_col='sample_id')
	X_test = test.iloc[:, 1:]

	for tfs in transformers:
		X_test = tfs.transform(X_test)

	y_test = np.argmax(model.predict(X_test), axis=1)

	df_subm = pd.DataFrame({'land_cover': y_test, 'sample_id': test.index})
	df_subm = df_subm[['sample_id', 'land_cover']]
	filename = f'TF_submission-{file_append}.csv'
	df_subm.to_csv(filename, index=False)

	print(f'Submission file ready (name={filename}, shape={df_subm.shape})')

if __name__ == "__main__":
	print('Start model training...')
	print('-'*20)

	X_train, X_val, y_train, y_val = load_data()
	
	scaler = PowerTransformer().fit(X_train)
	X_train_scaled, X_val_scaled = scaler.transform(X_train), scaler.transform(X_val)

	pca = PCA(n_components=300).fit(X_train_scaled)
	X_train_pca, X_val_pca = pca.transform(X_train_scaled), pca.transform(X_val_scaled)

	model = build_model(X_train_pca)
	
	f1 = train_model(model, X_train_pca, X_val_pca, y_train, y_val, save=False)

	write_submission_file(model, [scaler, pca], f1)