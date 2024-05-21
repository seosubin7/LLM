from transformers import pipeline
from langchain_core.runnables.base import Runnable
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"temperature": 0.7},
)


template = """Answer the question based only on the following context:\n

{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)



answer = llm.invoke("What are the sales goals?")
print(answer)