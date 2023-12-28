import os
import time
import pandas as pd

DATA_FILE = 'Greek_Parliament_Proceedings_1989_2020.csv'
DATA_FILE = 'sample_data.csv'
DATA_FOLDER = 'data'

# Get the data file path
current_path = os.getcwd()
csv_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), DATA_FILE)

# Using pandas & chunks
start_time = time.time()

chunksize = 100000
df = pd.read_csv(csv_file_path, chunksize=chunksize)
output_df = pd.DataFrame()
output_chunks = []
counter = 0

for chunk in df:
    grouped_chunk = chunk.groupby(['political_party', 'member_name', 'sitting_date'], as_index=False).agg({
        'parliamentary_period': 'first',
        'parliamentary_session': 'first',
        'parliamentary_sitting': 'first',
        'government': 'first',
        'member_region': 'first',
        'roles': 'first',
        'member_gender': 'first',
        'speech': ' '.join})
    output_chunks.append(grouped_chunk)

end_time = time.time()
print('Processing execution time: ', (end_time - start_time), 'sec')
start_time = time.time()

output_df = pd.concat(output_chunks, ignore_index=True)

# Save the output DataFrame to a CSV file
current_path = os.getcwd()
output_file_path = os.path.join(os.path.join(os.getcwd(), DATA_FOLDER), "output_file.csv")
output_df.to_csv(output_file_path, index=False)
end_time = time.time()
print('Saving execution time: ', (end_time - start_time), 'sec')
print(output_df.head())
