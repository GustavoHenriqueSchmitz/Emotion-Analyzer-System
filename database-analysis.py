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

# Filter out rows where the emotion is 'anticipation'
df = df[df['sentiment'] != 'relief']

# Save the modified DataFrame back to a CSV file
df.to_csv('./text_emotion.csv', index=False)

# Print the updated unique emotions and their counts
print("\nUpdated unique emotions found in the dataset:")
print("=====================================")
unique_emotions = df['sentiment'].unique()
for emotion in unique_emotions:
    count = df[df['sentiment'] == emotion].shape[0]
    print(f"emotion: {emotion} - Count: {count}")

print(f"\nTotal number of unique emotions: {len(unique_emotions)}")
