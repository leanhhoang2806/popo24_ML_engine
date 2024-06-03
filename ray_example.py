import ray
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Initialize Ray
ray.init(address='auto')

# Define a remote function for the GPT-2 generation task
@ray.remote
def generate_text(input_sequence):
    # For reproducibility
    SEED = 34
    tf.random.set_seed(SEED)

    # Maximum number of words in output text
    MAX_LEN = 70

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

    sample_output = GPT2.generate(
        input_ids, 
        do_sample=True, 
        max_length=MAX_LEN, 
        top_k=0, 
        temperature=0.8
    )

    return tokenizer.decode(sample_output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    input_sequence = "I don't know about you, but there's only one thing I want to do after a long day of work"
    
    # Call the remote function
    result = ray.get(generate_text.remote(input_sequence))

    print("Output:\n" + 100 * '-')
    print(result)
