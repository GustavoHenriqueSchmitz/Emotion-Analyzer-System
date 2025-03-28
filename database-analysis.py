import pandas as pd

# Read the CSV
df = pd.read_csv("./sentiment_analysis_dataset_filtered.csv")

# Count how many times each emotion appears
emotion_counts = df['emotion'].value_counts()

# Print the full count for each unique emotion
print(emotion_counts)

# If you want just one specific emotion, for example, "happiness":
specific_count = (df['emotion'] == 'happiness').sum()
print("Number of rows labeled 'happiness':", specific_count)
