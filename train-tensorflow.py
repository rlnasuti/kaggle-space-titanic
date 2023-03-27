import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Create new Cabin features
    cabin_split = data["Cabin"].str.split("/", expand=True)
    cabin_split.columns = ["CabinDeck", "CabinNum", "CabinSide"]
    data = pd.concat([data.drop("Cabin", axis=1), cabin_split], axis=1)
    
    # Fill missing values
    data["HomePlanet"].fillna(data["HomePlanet"].mode(), inplace=True)
    data["CryoSleep"].fillna(False, inplace=True)
    data["CabinDeck"].fillna(data["CabinDeck"].mode(), inplace=True)
    data["CabinSide"].fillna(data["CabinSide"].mode(), inplace=True)
    data["Destination"].fillna(data["Destination"].mode(), inplace=True)
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data["VIP"].fillna(False, inplace=True)
    data["RoomService"].fillna(data["RoomService"].median(), inplace=True)
    data["FoodCourt"].fillna(data["FoodCourt"].median(), inplace=True)
    data["ShoppingMall"].fillna(data["ShoppingMall"].median(), inplace=True)
    data["Spa"].fillna(data["Spa"].median(), inplace=True)
    data["VRDeck"].fillna(data["VRDeck"].median(), inplace=True)

    # Convert bools to ints
    data['CryoSleep'] = data['CryoSleep'].astype(int)
    data['VIP'] = data['VIP'].astype(int)
    
    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['HomePlanet'])
    data = pd.get_dummies(data, columns=['CabinDeck'])
    data = pd.get_dummies(data, columns=['CabinSide'])
    data = pd.get_dummies(data, columns=['Destination'])

    # Create IsYoung feature
    data['IsYoung'] = data['Age'] <= 15

    # Create TotalSpend feature
    data['TotalSpend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']


    # Feature scaling
    global features_to_scale
    features_to_scale = [
        "Age", "TotalSpend"
    ]

    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale]).astype(np.float32)

    data = data.drop(
        columns=
        [
            'Name', 
            'CabinNum', 
            'RoomService', 
            'FoodCourt', 
            'ShoppingMall', 
            'Spa', 
            'VRDeck'
        ]
    )

    return data

# Load the data
train_data: pd.DataFrame = pd.read_csv("data/train.csv")

train_data = preprocess_data(train_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(columns=['Transported']), train_data["Transported"], test_size=0.2, random_state=42)

# Define the model
model: tf.keras.Sequential = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(len(X_train.columns),)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

# Define the callback function to save the model with the best validation accuracy
checkpoint_path = "models/potential_best_sequentialnn_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# Train the model
history: tf.keras.callbacks.History = model.fit(
    X_train.astype(np.float32), 
    y_train.astype(np.float32), 
    epochs=500, 
    batch_size=32, 
    validation_data=(X_val.astype(np.float32), y_val.astype(np.float32)), 
    callbacks=[checkpoint]
    )


# Load the best saved model
best_model = tf.keras.models.load_model(checkpoint_path)

# Evaluate the model on the test data
test_loss, test_acc = best_model.evaluate(X_val.astype(np.float32), y_val.astype(np.float32))
print('Test accuracy:', test_acc)

# Load the current best test accuracy value from a file
accuracy_file = "models/test_accuracy.txt"
try:
    with open(accuracy_file, "r") as f:
        best_test_accuracy = float(f.read())
except FileNotFoundError:
    best_test_accuracy = 0

# Compare the current model's accuracy to the stored best accuracy
if test_acc > best_test_accuracy:
    # Save the current model
    model.save('models/best_sequentialnn_model.h5')

    # Update the stored best test accuracy value
    with open(accuracy_file, "w") as f:
        f.write(str(test_acc))

    print("The current model has a higher accuracy. It has been saved.")

    # Prepare a submission

    # Load the test data from 'data/test.csv'
    test_data = pd.read_csv("data/test.csv")

    # Save the passenger IDs for the submission file
    passenger_ids = test_data["PassengerId"]

    # Preprocess the test data
    X_test = preprocess_data(test_data)

    # Get predictions for the test data
    predictions = best_model.predict(X_test.astype(np.float32)).round().astype(bool).flatten()

    # Save the predictions to a file called 'submission.csv'
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": predictions})
    submission.to_csv("submission.csv", index=False)

