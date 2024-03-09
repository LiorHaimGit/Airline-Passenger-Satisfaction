# preprocess_data.py

from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Create a label encoder
    le = LabelEncoder()

    # Fit the encoder and transform the required columns
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Customer Type'] = le.fit_transform(data['Customer Type'])
    data['Type of Travel'] = le.fit_transform(data['Type of Travel'])
    data['Class'] = le.fit_transform(data['Class'])
    data['satisfaction'] = le.fit_transform(data['satisfaction'])

    # Drop unnecessary columns
    data = data.drop(columns=['number', 'id'])

    # Rename the target column
    data = data.rename(columns={'satisfaction': 'target'})

    return data
