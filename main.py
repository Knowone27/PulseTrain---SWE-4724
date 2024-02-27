import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def process_csv_files(input_directory="PulseTrainData"):
    # Get the current directory
    current_directory = os.getcwd()
    # Join the current directory with the specified subdirectory
    input_path = os.path.join(current_directory, input_directory)
    # Get a list of all CSV files in the specified subdirectory
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))

    # Dictionary to store processed DataFrames
    processed_data_frames = {}
    
    # Iterate through each CSV file
    for counter, file_path in enumerate(csv_files, start=1):
        # Process the CSV file and store the DataFrame in the dictionary
        processed_data_frames[f'data_frame_{counter}'] = process_csv_to_dataframe(file_path)

    # Return the dictionary of processed DataFrames
    return processed_data_frames

def process_csv_to_dataframe(input_file):
    # Read the CSV file into a DataFrame, skipping the first 4 rows and with no header
    df = pd.read_csv(input_file, skiprows=4, header=None)
    return df

def train_random_forest_model(X_train, y_train):
    # Initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    return clf

def main():
    # Process CSV files and obtain a dictionary of DataFrames
    processed_data_frames = process_csv_files()

    # read data frame
    target_data_frame_key = 'data_frame_10' # replace with data_frame_n
    
    # Check if the target DataFrame key exists in the processed dictionary
    if target_data_frame_key in processed_data_frames:
        # Retrieve the target DataFrame
        target_df = processed_data_frames[target_data_frame_key]

        # Example feature engineering (needs to be adjusted based on our data)
        # Here we're just using mean and standard deviation as features
        X = target_df.drop([4], axis=1)  # Assuming column 4 is the target
        y = target_df[4]  # Target column

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = train_random_forest_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()