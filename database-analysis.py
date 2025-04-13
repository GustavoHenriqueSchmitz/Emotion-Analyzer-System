import re
import emoji
import pandas as pd

code_instance = int(input("Which code instance do you wanna run [1/2/3/4/5/6/7/8]:"))

# Count all emotions in the dataset
if code_instance == 1:
    df = pd.read_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)

# Join CSV files into one
if code_instance == 2:
    csv_files = [
        "./Sentiment_Analysis_Data_Set_Backup/Dell_tweets_emotions.csv",
        "./Sentiment_Analysis_Data_Set_Backup/Emotion_classify_Data.csv",
        "./Sentiment_Analysis_Data_Set_Backup/Movies_Reviews.csv",
        "./Sentiment_Analysis_Data_Set_Backup/PoemDataset.csv",
        "./Sentiment_Analysis_Data_Set_Backup/text_emotion.csv"
    ]
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        df = df.iloc[:, [0, 1]]
        df.columns = ['text', 'emotion']
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv", index=False)

# Delete all emotions keeping only a specified amount
if code_instance == 3:
    df = pd.read_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv")
    unique_emotions = df['emotion'].astype(str).unique()
    processed_dfs = []
    for emotion in unique_emotions:
        emotion_subset = df[df['emotion'] == emotion]
        if len(emotion_subset) > 6500:
            emotion_subset = emotion_subset.sample(n=6500, random_state=42)
        processed_dfs.append(emotion_subset)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df.to_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv", index=False)

# Remove duplicated columns
if code_instance == 4:
    df = pd.read_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv")
    df_deduplicated = df.drop_duplicates(subset=['text'], keep='first')
    df_deduplicated.to_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv", index=False)

# Split the main dataset into training and testing datasets
if code_instance == 5:
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

# Clean the main dataset and save it back
if code_instance == 6:
    def clean_text(text):
        text = re.sub(r'@\w+', '<USER>', text)
        text = emoji.demojize(text, delimiters=('_', '_'))
        return text

    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    train_data['text'] = train_data['text'].astype(str).apply(clean_text)
    test_data['text'] = test_data['text'].astype(str).apply(clean_text)
    train_data.to_csv("./data/train.csv", index=False)
    test_data.to_csv("./data/test.csv", index=False)

# Delete all rows for a specific emotion
if code_instance == 7:
    emotion_to_delete = input("Enter the exact emotion name you want to delete: ").strip()
    df = pd.read_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv")
    df_filtered = df[df['emotion'] != emotion_to_delete].copy()
    df_filtered.to_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv", index=False)

# Convert 'emotion' column to lowercase
if code_instance == 8:
    df = pd.read_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv")
    df['emotion'] = df['emotion'].astype(str).str.lower()
    df.to_csv("./Sentiment_Analysis_Data_Set_Backup/sentiment_analysis.csv", index=False)
