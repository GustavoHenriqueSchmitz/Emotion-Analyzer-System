import pandas as pd

# Count all emotions in the dataset
df = pd.read_csv("./data/sentiment_analysis.csv")
emotion_counts = df['emotion'].value_counts()
print(emotion_counts)

# Join CSV files into one
csv_files = [
    "table1.csv",
    "table2.csv",
    "table3.csv",
    "table4.csv",
    "table5.csv"
]

dataframes = []

for file in csv_files:
    df = pd.read_csv(file, header=None)
    df = df.iloc[:, [0, 1]]
    df.columns = ['text', 'emotion']
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("sentiment_analysis.csv", index=False)

# Delete all emotions keeping only a specified amount
df = pd.read_csv("combined.csv")

emotion_df = df[df["emotion"] == "emotion"]
other_df = df[df["emotion"] != "emotion"]

if len(emotion_df) > 4000:
    emotion_df = emotion_df.sample(n=4000, random_state=42)
else:
    pass

final_df = pd.concat([other_df, emotion_df], ignore_index=True)
final_df.to_csv("sentiment_analysis.csv", index=False)