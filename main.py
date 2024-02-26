import os
import glob
import pandas as pd

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
        pd.set_option('display.max_rows', None) # remove this line if need only summary of rows

        # Get unique values from the 5th column, excluding the value 1
        unique_values = target_df[4].unique()
        unique_values = [value for value in unique_values if value != 1]

        # Print the unique values
        print("Unique Values:", unique_values)

        # Filter and print the DataFrame for each unique value in the 5th column
        for number in unique_values:
            filtered_df = target_df[target_df.iloc[:, 4] == number]
            print(f"Filtered data for {number}:\n", filtered_df)

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()