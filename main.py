import os
import glob
import pandas as pd

if __name__ == "__main__":
    
    # Get the current directory and append the "PulseTrainData" subdirectory
    current_directory = os.path.join(os.getcwd(), "PulseTrainData")
    # Get a list of all CSV files in the current directory
    csv_files = glob.glob(os.path.join(current_directory, "*.csv"))
    
    counter = 1
    # Loop through each CSV file
    for file_path in csv_files:
        # Open the file
        with open(file_path, "r") as file:
            # Read all lines in the file
            lines = file.readlines()
            # Skip the first 4 lines
            lines = lines[4:]
            
        # Create a new file path for the new file
        new_file_path = os.path.join(os.getcwd(), f"new_file_{counter}.csv")
        # Open the new file and write the lines to it
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(lines)
        # Increment the counter
        counter += 1
        
    # Get the current directory
    current_directory = os.getcwd()
    # Get a list of all CSV files in the current directory
    csv_files = glob.glob(os.path.join(current_directory, "*.csv"))

    # Read the CSV file, assuming it has no header
    data = pd.read_csv('new_file_10.csv', header=None)  # replace with your CSV file name

    # Get the unique values in the 5th column
    unique_values = data[4].unique()  # Indexing is 0-based, so the 5th column is at index 4

    # Convert the array to a list
    unique_values = list(unique_values)
    # Define the bad number
    bad_number = 1
    # If the bad number is in the list, remove it
    if bad_number in unique_values:
        unique_values.remove(bad_number)

    # Print the list of unique values
    print(unique_values)
        
    # Read the CSV file
    df = pd.read_csv('new_file_10.csv')  # Replace with your CSV file path

    # Filter rows and write to new CSV files
    for number in unique_values:
        # Get all rows where the 5th column equals the current number
        filtered_df = df[df.iloc[:, 4] == number]  # Assuming the fifth column is at index 4
        # Write these rows to a new CSV file
        filtered_df.to_csv(f'filtered_for_{number}.csv', index=False)