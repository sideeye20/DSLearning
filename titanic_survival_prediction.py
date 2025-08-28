#import matplotlib.pyplot as plt 
#import numpy as np
#x = [1, 2, 2.5, 3, 4]
#y = [1, 4, 7, 9, 15]
#plt.plot(x, y, 'ro')
#plt.axis([0, 6, 0, 20])
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
#plt.title('Linear Fit Example')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, Normalization,Input ,Concatenate, Dense 
from tensorflow.keras import Model


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') 
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
print(dftrain.head())
dftrain.describe()
print(dftrain.shape)
print(y_train.head())
dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% Survive')

#feature engineering
CATEGORICAL_COLUMNS = ['sex', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare','n_siblings_spouses','parch']

inputs={}
encoded_features = []

#categorical columns
for col in CATEGORICAL_COLUMNS:
    inputs[col] = Input(shape=(1,), name=col, dtype=tf.string)
    lookup = StringLookup(vocabulary=dftrain[col].astype(str).unique().tolist(),output_mode='one_hot')
    encoded = lookup(inputs[col])  
    encoded_features.append(encoded)

#numeric columns
for col in NUMERIC_COLUMNS: 
    inputs[col] = Input(shape=(1,), name=col)  
    norm = Normalization()
    # Fill missing values before adapting
    norm.adapt(np.array(dftrain[col].fillna(dftrain[col].mean()), dtype='float32').reshape(-1,1))  
    encoded = norm(inputs[col])
    encoded_features.append(encoded)

# Combine
feature_layer = Concatenate()(encoded_features)  

#dense layers
dense = Dense(128, activation='relu')(feature_layer)
output=Dense(1,activation='sigmoid')(dense)

#model
model = Model(inputs=inputs, outputs=output)  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

#convert DataFrame to dictionary format
train_dict = {}
eval_dict = {}

for col in inputs.keys():
    if col in NUMERIC_COLUMNS:
        # Fill missing values before converting to numpy arrays
        train_dict[col] = dftrain[col].fillna(dftrain[col].mean()).astype('float32').to_numpy().reshape(-1, 1)
        eval_dict[col] = dfeval[col].fillna(dftrain[col].mean()).astype('float32').to_numpy().reshape(-1, 1)
    else:
        train_dict[col] = dftrain[col].astype(str).to_numpy().reshape(-1, 1)  
        eval_dict[col] = dfeval[col].astype(str).to_numpy().reshape(-1, 1)


#train
model.fit(train_dict, y_train, epochs=10, batch_size=32, validation_data=(eval_dict, y_eval))  # fit the model to the training data and validate on the evaluation data

#evaluate
loss,accuracy = model.evaluate(eval_dict, y_eval, verbose=0)  # evaluate the model on the evaluation data
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Make predictions and plot
predictions = model.predict(eval_dict)  
print("Predictions: ", predictions)  # print the predictions
plt.figure(figsize=(10, 6))  # create a figure for plotting
plt.scatter(range(len(predictions)), predictions, color='blue', label='Predictions')  # scatter plot of predictions
plt.title('Predictions vs Index')  # set the title of the plot
plt.xlabel('Index')  # set the x-axis label
plt.ylabel('Predicted Probability')  # set the y-axis label
plt.legend()  # show the legend
plt.show()  # display the plot

# Save model
model.save('titanic_model.keras')
