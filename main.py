import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# Function to process CSV files and return a dictionary of DataFrames
def process_csv_files(input_directory="PulseTrainData"):

    current_directory = os.getcwd()

    input_path = os.path.join(current_directory, input_directory)

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))

    processed_data_frames = {}

    interleaved_counter = 1
    
    # Iterate through each CSV file
    for file_path in csv_files:
        df = process_csv_to_dataframe(file_path)

        if "Interleaved" in file_path:
            key = f'Interleaved_{interleaved_counter}'
            interleaved_counter += 1
        else:
            key = f'data_frame_{interleaved_counter}'
            interleaved_counter += 1
            
        processed_data_frames[key] = df

    return processed_data_frames

#Read the CSV file into a DF and skip first 4 rows and BUFFERs
def process_csv_to_dataframe(input_file):
    print(f"Processing file: {input_file}")
    df = pd.read_csv(input_file, skiprows=4, header=None)

    df_filtered = df[~df.iloc[:, 0].str.contains('BUFFER')]

    return df_filtered

# Preprocessing function to reshape DataFrame into a suitable shape for the CNN
def preprocess_data(df):

    print("DataFrame before preprocessing:\n", df)

    df_filtered = df[df[4] != 1.0]
    
    num_samples = df_filtered.shape[0]  
    print("Number of samples in the DataFrame after filtering:", num_samples) 
    
    expected_size = num_samples * 20 * 20 * 1  
    print("Expected size after reshaping:", expected_size)  
    
    # Resize the DataFrame to evenly divide the expected size after reshaping
    resize_factor = expected_size // df_filtered.size
    resized_df = df_filtered.iloc[np.repeat(np.arange(len(df_filtered)), resize_factor)]
    
    # Reshape the resized DataFrame into a 20x20 grid with a single channel
    df_as_grid = resized_df.values.reshape(num_samples, 20, 20, 1)
    
    print("Reshaped DataFrame shape:", df_as_grid.shape) 

    return df_as_grid

# Convert unique angle values to categorical labels
def convert_to_categorical(df):
    # Filter the DataFrame to exclude rows where the Angle has a value of 1 (BUFFER had this but may not be needed now?)
    filtered_df = df[df[4] != 1]
    
    labels = to_categorical(filtered_df[4].values).astype(int)
    
    return labels

# CNN
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
    """
    This calls the process_csv_files function to load CSV files from the PulseTrainData directory and preprocess them into a dictionary of DataFrames (processed_data_frames).
    """
    processed_data_frames = process_csv_files()

    total_test_loss = 0
    total_test_accuracy = 0
    num_files_processed = 0

    # Iterate through each processed DataFrame
    for key, df in processed_data_frames.items():
        print({key})
        if key.startswith("Interleaved"):
            print(f"\nProcessing DataFrame from file: {key}")

            # Print all the unique values in each dataframe
            unique_values = df[4].unique()
            print(f"Unique Values in {key}:", unique_values)

            for number in unique_values:
                filtered_df = df[df.iloc[:, 4] == number]
                print(f"Filtered data for {number}:\n", filtered_df)

            preprocessed_data = preprocess_data(df)
            categorical_labels = convert_to_categorical(df)
            

            # Train and evaluate the model
            X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, categorical_labels, test_size=0.2, random_state=42)

            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)

            input_shape = X_train.shape[1:]
            num_classes = y_train.shape[1]
            model = create_cnn_model(input_shape, num_classes)
            model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
            score = model.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])


            # Accumulate the Scores
            total_test_loss += score[0]
            total_test_accuracy += score[1]
            num_files_processed += 1

    if num_files_processed > 0:
        average_test_loss = total_test_loss / num_files_processed
        average_test_accuracy = total_test_accuracy / num_files_processed
        print("\nOverall Test Loss:", average_test_loss)
        print("Overall Test Accuracy:", average_test_accuracy)
    else:
        print("No files processed.")

    print("\nAll Processing Finished.\n")

if __name__ == "__main__":
    main()
