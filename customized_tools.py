
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
)


print(calculator.description)
print(calculator.run({'a':2,'b':2}))


from langchain.agents import create_tool_calling_agent
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain import hub

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_path = "./hf_models/microsoft/Phi-3-mini-4k-instruct/"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#max_length has typically been deprecated for max_new_tokens 
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, model_kwargs={"temperature":0}
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt = hub.pull("hwchase17/openai-functions-agent")


agent = create_tool_calling_agent(llm, [calculator], prompt)




