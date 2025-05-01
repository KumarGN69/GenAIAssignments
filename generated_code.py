import pandas as pd
from PIL import Image
import io
import os

def read_and_remove_duplicates(file_name):
    try:
        # Read csv file into a DataFrame
        df = pd.read_csv(file_name)
        
        # Get duplicate rows
        duplicates = df[df.duplicated()]
        
        print("Duplicate records:")
        print(duplicates)
        
        # Remove duplicate rows from the original DataFrame
        updated_df = df.drop_duplicates()
        
        # Save the updated DataFrame to a new csv file
        updated_df.to_csv('updated_records.csv', index=False)
    
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except Exception as e:
        print(str(e))

def main():
    file_name = input("Please enter the name of your csv file: ")
    read_and_remove_duplicates(file_name)

if __name__ == "__main__":
    main()