import numpy as np
from tqdm import tqdm
import tensorflow as tf


np.random.seed(42)


def basicTrainingModel():
    inputs = np.random.rand(5, 5)
    outputs = np.random.rand(5, 1)  # Corrected to (5, 1)

    inputSize = inputs.shape[1]
    outputSize = outputs.shape[1]
    hiddenSize = 4
    learningRate = 0.01
    epochs = 10000

    # Initialize weights and biases
    W1 = np.random.rand(inputSize, hiddenSize)  # 5 x 4 matrix
    b1 = np.random.rand(1, hiddenSize)

    outputHidden = np.random.rand(hiddenSize, outputSize)
    outputBias = np.random.rand(1, outputSize)

    def activation(x):
        return np.maximum(0, x)

    def activation_derivative(x):
        return np.where(x > 0, 1, 0)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        # Forward pass
        W1_output = np.dot(inputs, W1) + b1
        W1_output_activation = activation(W1_output)

        prediction_output = np.dot(W1_output_activation, outputHidden) + outputBias
        prediction_activation = activation(prediction_output)

        # Loss function
        loss = 0.5 * np.sum((outputs - prediction_activation) ** 2)

        # Backward pass
        output_error = outputs - prediction_activation
        output_delta = output_error * activation_derivative(prediction_output)

        hidden_output_error = output_delta.dot(outputHidden.T)
        hidden_delta = hidden_output_error * activation_derivative(W1_output)

        # Update weights and biases
        outputHidden += W1_output_activation.T.dot(output_delta) * learningRate
        W1 += inputs.T.dot(hidden_delta) * learningRate
        outputBias += np.sum(output_delta, axis=0, keepdims=True) * learningRate
        b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learningRate

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Test the neural network
    hidden_layer_input = np.dot(inputs, W1) + b1
    hidden_layer_output = activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, outputHidden) + outputBias
    predicted_output = activation(output_layer_input)
    predicted_loss = 0.5 * np.sum((predicted_output - outputs) ** 2)

    print("Predicted output:")
    print(predicted_output)

    print("Predicted losss")
    print(predicted_loss)


def usingTensorFlow():
    np.random.seed(42)
    inputs = np.random.rand(5, 5).astype(np.float32)
    outputs = np.random.rand(5, 1).astype(np.float32)

    input_size = inputs.shape[1]
    first_layer = 6 # number of features to extract from a single data points from this layer
    second_layer = 4
    output_size = 1

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                first_layer, activation="relu", input_shape=(input_size,)
            ),
            tf.keras.layers.Dense(second_layer, activation="relu"),
            tf.keras.layers.Dense(output_size),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mean_squared_error"
    )

    # Training   the model with progress bar
    epochs = 10000
    batch_size = 5

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        model.train_on_batch(inputs, outputs)
        if epoch % 1000 == 0:
            loss = model.evaluate(inputs, outputs, verbose=0)
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Test the neural network
    predicted_output = model.predict(inputs)
    print("Predicted output:")
    print(predicted_output)


def usingTensorFlowGPU():
    # Ensure TensorFlow is using the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available, using CPU.")

    np.random.seed(42)
    inputs = np.random.rand(5, 5).astype(np.float32)
    outputs = np.random.rand(5, 1).astype(np.float32)

    input_size = inputs.shape[1]
    first_layer = 6 # number of features to extract from a single data points from this layer
    second_layer = 4
    output_size = 1

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                first_layer, activation="relu", input_shape=(input_size,)
            ),
            tf.keras.layers.Dense(second_layer, activation="relu"),
            tf.keras.layers.Dense(output_size),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mean_squared_error"
    )

    # Training the model with progress bar
    epochs = 10000
    batch_size = 5

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        model.train_on_batch(inputs, outputs)
        if epoch % 1000 == 0:
            loss = model.evaluate(inputs, outputs, verbose=0)
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Test the neural network
    predicted_output = model.predict(inputs)
    print("Predicted output:")
    print(predicted_output)


def runAll():
    # print("Running Basic Training Model")
    # basicTrainingModel()
    # print("============================")
    # print("")

    print("Running with tensorflow module with GPU")
    # usingTensorFlowGPU()
    print("============================")
    print("")


# runAll()


import pandas as pd
import os
from transformers import BertTokenizer, TFBertModel
# /home/hoang2/Documents/work/cheaper-ML-training/data/Fake.csv

current_directory = os.getcwd()

def get_max_length(titles, tokenizer):
    max_length = 0
    for title in titles:
        tokens = tokenizer.encode(title, truncation=False)
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length

def tokenized_text_columns(titles, tokenizer, bert_model, max_length):
    inputs = tokenizer(titles, padding='max_length', truncation=True, max_length=max_length, return_tensors="tf")
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.numpy()

def training(X, targets, model=None):
    if model is None:
        # Define the model
        input_shape = X.shape[1:]
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    # Train the model incrementally
    model.fit(X, targets, epochs=1, batch_size=32, validation_split=0.2)
    
    return model

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available, using CPU.")
    print("Running training process for fake news or not")
    print("Load Bert token Models")
    true = 1
    false = 0

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    current_directory = os.getcwd()
    true_file_path = os.path.join(current_directory, 'data/True.csv')
    fake_file_path = os.path.join(current_directory, 'data/Fake.csv')

    df_true = pd.read_csv(true_file_path)
    df_fake = pd.read_csv(fake_file_path)

    df_true['target'] = 1
    df_fake['target'] = 0

    # Concatenate the DataFrames
    df_combined = pd.concat([df_true, df_fake], ignore_index=True)

    # Shuffle the combined DataFrame
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    # Determine the maximum length of tokens
    max_length = get_max_length(df_combined['title'].tolist(), tokenizer)
    print(f"Determined maximum token length: {max_length}")

    batch_size = 100
    num_batches = len(df_combined) // batch_size

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(63, 768)), # these data is injected from trial and error
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    for i in tqdm(range(num_batches + 1), desc="Training Progress"):
        batch_df = df_combined[i * batch_size: (i + 1) * batch_size]
        if batch_df.empty:
            continue

        tokenized_texts = tokenizer(batch_df['title'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors="tf")
        bert_outputs = bert_model(**tokenized_texts)

        embeddings = bert_outputs.last_hidden_state

        embeddings_list = embeddings.numpy()
        targets = batch_df['target'].values

        model.fit(embeddings_list, targets, epochs=10, batch_size=32, validation_split=0.2)
        # break
    
    # Evaluate the model
    validation = df_combined.iloc[:100]
    tokenized_texts = tokenizer(validation['title'].tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors="tf")
    bert_outputs = bert_model(**tokenized_texts)
    embeddings = bert_outputs.last_hidden_state.numpy()
    targets = validation['target'].values


    (loss, accuracy) = model.evaluate(embeddings, targets, verbose=0)
    print("Evaluation Metrics:")
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
if __name__ == "__main__":
    main()













# import ray
# from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
# import tensorflow as tf

# # Initialize Ray
# ray.init(address='auto')

# # Define a remote function for the GPT-2 generation task
# @ray.remote
# def generate_text(input_sequence):
#     # For reproducibility
#     SEED = 34
#     tf.random.set_seed(SEED)

#     # Maximum number of words in output text
#     MAX_LEN = 70

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#     input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

#     sample_output = GPT2.generate(
#         input_ids, 
#         do_sample=True, 
#         max_length=MAX_LEN, 
#         top_k=0, 
#         temperature=0.8
#     )

#     return tokenizer.decode(sample_output[0], skip_special_tokens=True)

# # Example usage
# if __name__ == "__main__":
#     input_sequence = "I don't know about you, but there's only one thing I want to do after a long day of work"
    
#     # Call the remote function
#     result = ray.get(generate_text.remote(input_sequence))

#     print("Output:\n" + 100 * '-')
#     print(result)
