import pandas as pd

def convert_txt_to_csv_pandas(input_filename, output_filename):
    """
    Converts a text file with CSV-like data to a CSV file using pandas.

    Args:
        input_filename (str): The path to the input text file.
        output_filename (str): The path where the CSV file will be saved.
    """
    try:
        # Read the text file into a pandas DataFrame
        # pandas is smart enough to infer the delimiter if it's comma-separated
        df = pd.read_csv(input_filename)
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_filename, index=False) # index=False prevents pandas from writing the DataFrame index as a column
        
        print(f"Successfully converted '{input_filename}' to '{output_filename}' using pandas.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use it ---

for i in range(1, 5):
    input_file = f"drone_trajectory_scenario_{i}.txt"
    output_file = f"drone_trajectory_scenario_{i}.csv"
    convert_txt_to_csv_pandas(input_file, output_file)
