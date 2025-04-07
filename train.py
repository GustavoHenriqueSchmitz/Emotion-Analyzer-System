from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

VOCAB_SIZE = 10000  # Max words in vocabulary
SEQUENCE_LENGTH = 500 # Max words per text sample
EMBEDDING_DIM = 128 # Dimension for word embeddings
LSTM_UNITS = 64     # Units in the LSTM layer

# --- 1. Getting the data ---
try:
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Make sure paths like './data/train.csv' are correct.")
    exit()

# --- 2. Preprocessing the data ---
# Select the text column and ensure it's string type
x_train_text = train_data['Text'].astype(str).values
x_test_text = test_data['Text'].astype(str).values

# Create and adapt the TextVectorization layer
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=SEQUENCE_LENGTH)
vectorize_layer.adapt(x_train_text) # Adapt ONLY on training text

# Vectorize the text
x_train = vectorize_layer(x_train_text)
x_test = vectorize_layer(x_test_text)

# Convert categorical variable into dummy/indicator variables (One-Hot Encoding)
y_train_labels = train_data['emotion']
y_test_labels = test_data['emotion']

y_train = pd.get_dummies(y_train_labels)
y_test = pd.get_dummies(y_test_labels)

# --- 3. Create the model ---
# Model architecture suitable for text sequences
model = keras.Sequential([
    keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True),
    keras.layers.LSTM(LSTM_UNITS),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes = y_train.shape[1], activation='softmax')
])

# --- 4. Compile the model ---
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.build(input_shape=(None, SEQUENCE_LENGTH)) # Build the model explicitly to see summary
model.summary() # Print model structure

# --- 5. Train the model ---
print("\n--- Starting Training ---")
# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Use validation_data and ADD the callback
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)
print("--- Training Complete ---")

# --- 6. Evaluate the model on the test data ---
print("\n--- Evaluating Model ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# --- 7. Print the test results ---
print("================= Test Results ==================")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print("=================================================")

# --- 8. Save the model ---
while True:
    answer = input("According to the results, do you want to continue and save the model for use [y/n]: ").lower()
    if answer not in ["y", "n"]:
        print("Please enter a valid answer.")
    elif answer == "y":
        print("Saving Model...")
        model.save('Sentiment_Analysis_model.keras')
        print("Model saved.")
        break
    else:
        print("Model not saved.")
        break
    