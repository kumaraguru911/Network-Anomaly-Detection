import pandas as pd
import os

# Read the original data file
print("Reading original data file...")
data_file = 'data/data.csv'

if os.path.exists(data_file):
    # Read data
    df = pd.read_csv(data_file)
    
    # Strip whitespace from column names to prevent KeyErrors
    df.columns = df.columns.str.strip()
    
    # Display info about the data
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Label distribution:")
    print(df['Label'].value_counts())
    
    # Create a test dataset (20% of original data or max 10,000 rows)
    sample_size = min(int(len(df) * 0.2), 10000)
    test_df = df.sample(n=sample_size, random_state=42)
    
    # Save as test data
    test_file = 'data/test_data.csv'
    test_df.to_csv(test_file, index=False)
    print(f"Test dataset created with {len(test_df)} samples")
    print(f"Saved to {test_file}")
else:
    print(f"Error: Cannot find {data_file}")