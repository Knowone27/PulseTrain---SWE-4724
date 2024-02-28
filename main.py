import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

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

# Preprocessing function to reshape DataFrame into a suitable shape for the CNN
def preprocess_data(df):
    # Filter out rows where the fifth column has a value of 1
    df_filtered = df[df[4] != 1]
    
    num_samples = df_filtered.shape[0]  # Get the number of samples (rows) in the filtered DataFrame
    print("Number of samples in the DataFrame after filtering:", num_samples)  # Print the number of samples
    
    # Define the expected size for reshaping based on the size of the filtered DataFrame
    expected_size = num_samples * 20 * 20 * 1  
    print("Expected size after reshaping:", expected_size)  # Print the expected size
    
    # Resize the DataFrame to evenly divide the expected size after reshaping
    resize_factor = expected_size // df_filtered.size
    resized_df = df_filtered.iloc[np.repeat(np.arange(len(df_filtered)), resize_factor)]
    
    # Reshape the resized DataFrame into a 20x20 grid with a single channel
    df_as_grid = resized_df.values.reshape(num_samples, 20, 20, 1)
    
    print("Reshaped DataFrame shape:", df_as_grid.shape)  # Print the shape of the reshaped DataFrame

    return df_as_grid

# Convert unique angle values to categorical labels
def convert_to_categorical(df):
    # Filter the DataFrame to exclude rows where the fifth column has a value of 1
    filtered_df = df[df[4] != 1]
    
    # Convert unique angle values from the filtered DataFrame to categorical labels
    labels = to_categorical(filtered_df[4].values)
    
    return labels

# CNN architecture
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def main():
    # Process CSV files and obtain a dictionary of DataFrames
    processed_data_frames = process_csv_files()

    # read data frame
    target_data_frame_key = 'data_frame_10' # replace with data_frame_n
    
    # Check if the target DataFrame key exists in the processed dictionary
    if target_data_frame_key in processed_data_frames:
        # Retrieve the target DataFrame
        target_df = processed_data_frames[target_data_frame_key]

        # Set the option to display all rows
        # pd.set_option('display.max_rows', None) # remove this line if need only summary of rows

        # Get unique values from the 5th column, excluding the value 1
        unique_values = target_df[4].unique()
        unique_values = [value for value in unique_values if value != 1]

        # Print the unique values
        print("Unique Values:", unique_values)

        # Filter and print the DataFrame for each unique value in the 5th column
        for number in unique_values:
            filtered_df = target_df[target_df.iloc[:, 4] == number]
            print(f"Filtered data for {number}:\n", filtered_df)

    # Preprocess the DataFrame
    preprocessed_data = preprocess_data(target_df)
    categorical_labels = convert_to_categorical(target_df)

    # Convert input data to float32
    preprocessed_data = preprocessed_data.astype(np.float32)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, categorical_labels, test_size=0.2, random_state=42)

    # Define input shape based on the preprocessed data shape
    input_shape = X_train.shape[1:]

    # Define the number of unique classes based on the shape of the labels
    num_classes = y_train.shape[1]

    # Create the CNN model
    model = create_cnn_model(input_shape, num_classes)

    # Train the CNN model
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

    # Evaluate the CNN model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Use the model to make predictions
    predictions = model.predict(X_test)

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
