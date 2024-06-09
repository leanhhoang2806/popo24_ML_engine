import os
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling
from tqdm import tqdm

def main():
    # Set up distributed training environment
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Check if GPU is available and print device information
    if tf.config.experimental.list_physical_devices('GPU'):
        print(f"CUDA is available. Using GPU: {tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])['device_name']}")
    else:
        print("CUDA is not available. Using CPU.")

    with strategy.scope():
        # Step 1: Prepare the Text Data
        file_path = '/app/data/legal-guide-for-starting-and-running-a-small-business.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()

        def preprocess_text(text):
            text = text.lower()
            # Add more preprocessing steps if necessary
            return text

        text_data = preprocess_text(text_data)

        # Step 2: Tokenize the Text
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(text_data, add_special_tokens=True)

        # Split tokens into manageable chunks
        chunk_size = 10000  # Adjust this size based on your GPU memory capacity
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Create a dataset from the tokens
        def create_dataset(tokens_chunk, block_size=128):
            def generate_examples(tokens, block_size=128):
                for i in range(0, len(tokens) - block_size + 1, block_size):
                    yield tokens[i:i + block_size], tokens[i + 1:i + block_size + 1]

            token_dataset = tf.data.Dataset.from_generator(
                lambda: generate_examples(tokens_chunk),
                output_types=(tf.int32, tf.int32),
                output_shapes=((128,), (128,))
            )

            token_dataset = token_dataset.shuffle(buffer_size=1000).batch(2)
            return token_dataset

        # Create a data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Step 3: Fine-tune the GPT-2 Model
        model = TFGPT2LMHeadModel.from_pretrained('gpt2')
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        # Adjust batch size based on the number of workers
        global_batch_size = 2 * strategy.num_replicas_in_sync
        batch_size_per_worker = 2
        batch_size = batch_size_per_worker * strategy.num_replicas_in_sync

        # Gradient accumulation
        accumulate_gradients_every = 4  # Accumulate gradients over 4 batches

        def distributed_train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True).logits
                per_example_loss = loss_fn(y_batch, logits)
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        for tokens_chunk in token_chunks:
            dataset = create_dataset(tokens_chunk)
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            steps_per_epoch = len(tokens_chunk) // 128 // global_batch_size
            total_steps = 1 * steps_per_epoch
            progress_bar = tqdm(total=total_steps, desc="Training Progress")

            for epoch in range(1):

                for step, (x_batch, y_batch) in enumerate(dist_dataset):
                    distributed_loss = strategy.run(distributed_train_step, args=(x_batch, y_batch))

                    if step % 100 == 0:
                        progress_bar.set_postfix(loss=tf.reduce_mean(distributed_loss).numpy())

                    progress_bar.update(1)

                progress_bar.close()

    # Step 4: Generate Responses
    def generate_response(prompt, model, tokenizer, max_length=100):
        input_ids = tokenizer.encode(prompt, return_tensors='tf')
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    # Example usage of the chatbot
    prompt = "How to start a small business legally?"
    response = generate_response(prompt, model, tokenizer)
    print(response)

if __name__ == "__main__":
    main()
