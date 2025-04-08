import re
import emoji
import pandas as pd

code_instance = int(input("Which code instance do you wanna run [1/2/3/4]:"))

# Count all emotions in the dataset
if code_instance == 1:
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
    combined_df.to_csv("./data/sentiment_analysis.csv", index=False)

# Delete all emotions keeping only a specified amount
if code_instance == 2:
    df = pd.read_csv("./data/sentiment_analysis.csv")

    emotion_df = df[df["emotion"] == "emotion"]
    other_df = df[df["emotion"] != "emotion"]
    if len(emotion_df) > 4000:
        emotion_df = emotion_df.sample(n=4000, random_state=42)
    final_df = pd.concat([other_df, emotion_df], ignore_index=True)
    final_df.to_csv("./data/sentiment_analysis.csv", index=False)

# Split the main dataset into training and testing datasets
if code_instance == 3:
    df = pd.read_csv("./data/sentiment_analysis.csv")
    emotions = df["emotion"].unique()

    test_dfs = []
    train_dfs = []
    for emotion in emotions:
        emotion_subset = df[df["emotion"] == emotion]
        test_part = emotion_subset.sample(n=500, random_state=42)
        train_part = emotion_subset.drop(test_part.index)

        test_dfs.append(test_part)
        train_dfs.append(train_part)

    test_df = pd.concat(test_dfs, ignore_index=True)
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df.to_csv("./data/test.csv", index=False)
    train_df.to_csv("./data/train.csv", index=False)

# ! Didn´t executed this block yet, better prepare a backup before doing it
# Clean the main dataset and save it back
if code_instance == 4:
    def clean_text(text):
        text = re.sub(r'@\w+', '', text).strip()
        text = emoji.demojize(text, delimiters=('_', '_'))
        return text

    df = pd.read_csv("./data/sentiment_analysis.csv")
    df['text'] = df['text'].astype(str).apply(clean_text)
    df.to_csv("./data/sentiment_analysis.csv", index=False)
