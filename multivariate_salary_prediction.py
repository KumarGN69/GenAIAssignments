
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import Dense

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#
#----------function to encode different string values to numeric valid
def label_encode_column(column):
    le = LabelEncoder()
    return le.fit_transform(column)



#-------------Read the input file to dataframe--------------
df = pd.read_csv('./inputs/HR_Data.csv')

# #-----------drop duplicates and encode the categorical values------------------------
df = df.drop_duplicates()
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).apply(label_encode_column)

#------------create the sub data frames for features and dependent variables------------
inputx = df.drop('Monthly Income',axis=1)
inputy = df['Monthly Income']

#------------split into training and test data sets------------
input_train, input_test, output_train, output_test = train_test_split(inputx, inputy, test_size=1 / 3, random_state=42)

#----------------------scale the input train and test data sets---------------------
scaler_X = StandardScaler()
input_scaled_train = scaler_X.fit_transform(input_train)
input_scaled_test = scaler_X.transform(input_test)

#----------------------using PCA to find the right number of components----------------------
pca = PCA().fit(input_scaled_train)  # Fit PCA to your data X
cumulative_variance = pca.explained_variance_ratio_.cumsum()
n_components = (cumulative_variance < 0.95).sum() + 1
print(n_components)

#----------------------------create a PCA model for use with right set of components----------------------
pca_for_modelling = PCA(n_components=n_components)
input_scaled_train_reduced = pca_for_modelling.fit_transform(input_scaled_train)
input_scaled_test_reduced = pca_for_modelling.transform(input_scaled_test)

#----------------------scale the dependent variable train and test data sets---------------------
scaler_Y = StandardScaler()
output_scaled_train = scaler_Y.fit_transform(output_train.to_numpy().reshape(-1,1))
output_scaled_test = scaler_Y.transform(output_test.to_numpy().reshape(-1,1))

#-------------------instantiate a ANN model with reduced number of features --------------
model = Sequential()
model.add(Dense(54, input_dim=n_components, activation='relu'))
model.add(Dense(41, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --------------train the ANN model-------------
model.fit(input_scaled_train_reduced, output_scaled_train, epochs=300, batch_size=2, validation_data=(input_scaled_test_reduced, output_scaled_test))
loss, mse = model.evaluate(input_scaled_test_reduced, output_scaled_test)
print(model.metrics_names)
print(f"loss : {loss}, mse: {mse}")

#---------------predict using the model-----------
samples = inputx.iloc[56:57, ].to_numpy()
print(samples)
samples_scaled = scaler_X.transform(samples)
samples_scaled_reduced= pca_for_modelling.transform(samples_scaled)
#
pred_scaled_output = model.predict(samples_scaled_reduced)
predictions = scaler_Y.inverse_transform(pred_scaled_output)

print(f"predictions :{predictions}")


# #---------------------save the model in json format-------------------------------------
model_json = model.to_json()
with open("multivariate_salary_regression_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("multivariate_salary_regression_model.weights.h5")