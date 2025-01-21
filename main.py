%%writefile main.py
import textwrap
from preprocessing import preprocess_document
from retrieval import retrieve_relevant_chunks
from generation import generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Preprocess the document
file_path = "/content/Test.txt"
paragraphs = preprocess_document(file_path)

# Step 2: Split document into manageable chunks
chunk_size = 500
documents = textwrap.wrap(' '.join(paragraphs), width=chunk_size)

# Step 3: Define query and retrieve relevant chunks
query = "What are the applications of artificial intelligence in healthcare?"
top_chunks = retrieve_relevant_chunks(documents, query, top_k=5)

# Step 4: Load pre-trained language model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 5: Combine retrieval and generation
combined_context = ' '.join([chunk for chunk, _ in top_chunks])
response = generate_response(query, combined_context, model, tokenizer)

# Output
print("Query:", query)
print("\nGenerated Response:", response)
