import os
import time
import pandas as pd
import sys

def create_files(data_folder, data_file, output_file, output_sample_file, fraction=0.1):
    print('Processing the data...', file=sys.stderr)
    csv_file_path = os.path.join(os.path.join(os.getcwd(), data_folder), data_file)
    
    # Using pandas & chunks
    start_time = time.time()

    chunksize = 100000
    df = pd.read_csv(csv_file_path, chunksize=chunksize)
    output_df = pd.DataFrame()
    output_chunks = []

    for chunk in df:
        grouped_chunk = chunk.groupby(['political_party', 'member_name', 'sitting_date'], as_index=False).agg({
            'parliamentary_period': 'first',
            'parliamentary_session': 'first',
            'parliamentary_sitting': 'first',
            'government': 'first',
            'member_region': 'first',
            'roles': 'first',
            'member_gender': 'first',
            'speech': ' | '.join})
        output_chunks.append(grouped_chunk)

    end_time = time.time()
    print('Processing execution time: ', (end_time - start_time), 'sec', file=sys.stderr)
    start_time = time.time()

    output_df = pd.concat(output_chunks, ignore_index=True)
    sample_df = output_df.sample(frac = fraction)

    # Save the output DataFrame to a CSV file
    output_file_path = os.path.join(os.path.join(os.getcwd(), data_folder), output_file.format(data_file))
    output_sample_path = os.path.join(os.path.join(os.getcwd(), data_folder), output_sample_file.format(data_file))

    output_df.to_csv(output_file_path, index=False)
    sample_df.to_csv(output_sample_path, index=False)

    end_time = time.time()
    print('Saving execution time: ', (end_time - start_time), 'sec', file=sys.stderr)
    return len(output_df), len(sample_df)
