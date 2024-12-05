import pandas as pd


# Replace 'input_file.xlsx' with the name of your Excel file
excel_file = 'emg_signals_vvvv.xlsx'

# Replace 'output_file.csv' with the desired name for your CSV file
csv_file = 'EMG_data.csv'

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(excel_file)

# Save the DataFrame to a CSV file
df.to_csv(csv_file, index=False)
