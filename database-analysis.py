import pandas as pd

# Read the CSV file
df = pd.read_csv('./text_emotion.csv')

# Print the column names
print("\nAvailable columns in the dataset:")
print("================================")
print(df.columns.tolist())

# Get the first few rows to see the data structure
print("\nFirst few rows of the dataset:")
print("============================")
print(df.head())

# Get unique values in the Emotion column
unique_emotions = df['sentiment'].unique()

# Print the unique emotions and their counts
print("\nUnique emotions found in the dataset:")
print("=====================================")
for emotion in unique_emotions:
    count = df[df['sentiment'] == emotion].shape[0]
    print(f"emotion: {emotion} - Count: {count}")

print(f"\nTotal number of unique emotions: {len(unique_emotions)}")
