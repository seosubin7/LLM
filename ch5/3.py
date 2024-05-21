from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM


# Model
model_name_or_path = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=True)

# Pipeline
logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

# Define a simple function to generate a response
def generate_response(prompt):
    response = llm(prompt)
    return response

# Example usage
question = "What is the capital of France?"
response = generate_response(question)
print("Question:", question)
print("Response:", response)
