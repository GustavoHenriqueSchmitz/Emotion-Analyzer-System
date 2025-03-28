import pandas as pd

# Read the CSV
df = pd.read_csv("./sentiment_analysis_dataset.csv")

# Count how many times each emotion appears
emotion_counts = df['emotion'].value_counts()

# Print the full count for each unique emotion
print(emotion_counts)
