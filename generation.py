%%writefile generation.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_input(query, retrieved_context):
    return f"Answer Only the provided query without any comments or more words\nContext: {retrieved_context}\nQuery: {query}\nResponse:"

def generate_response(query, retrieved_context, model, tokenizer, max_length=500):
    formatted_input = format_input(query, retrieved_context)
    inputs = tokenizer(formatted_input, return_tensors="pt")
    output = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,
        temperature=0.7,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
