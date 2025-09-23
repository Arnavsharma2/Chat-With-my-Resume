from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

# load env vars
load_dotenv()

# assigning LLM model
llm = ChatOpenAI(
    model="gpt-3.5-turbo", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# path to the resume 
pdf_path = "/Users/aps/PythonProject/Chat-With-my-Resume/RAG Resume.pdf"


# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# load pdf assign to data
pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# chunking of data to improve accuracy also reduces hallucinations
# Smaller chunks = fewer API calls = lower cost
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# applies the chunking settings
pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

persist_directory = r"/Users/aps/PythonProject/Chat-With-my-Resume/VectorDB"
collection_name = "arnav_resume"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# creating vector data base with parameters
try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# retriever extracts relevant info from my vector DB when called
retriever = vectorstore.as_retriever(
    # search type similarity
    search_type = 'similarity',
    # search top k similarity in vector DB
    search_kwargs = {'k': 5} 
)

# @ tool makes it callable by LLM
@tool
# retriever tool activates retriever, input query
def retriever_tool(query: str) -> str:
    # Docstring to tell LLM what this program does
    """
    This tool searches and returns the information from the Arnav Resume document.
    """
    # sends query to retriever to get top k outputs that answer the prompt from vector DB, assigns it to docs
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the Arnav Resume document."
    
    results = []
    # prints out each of the top k outputs
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

# Tells LLM this tool is available / assigns it to list of LLM avail tools
tools = [retriever_tool]

llm = llm.bind_tools(tools)

# TypedDict, returns a dict of specified type, Annotated to add messages onto the original message, Sequence to make the messages a list
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# This checks to see if the user typed in another message, thus it needs a new output to be generated, it needs to continue | we check the last message for a tool call
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# Prompting the Artificial Intelligence properly, this is a very important step
system_prompt = """
You are an intelligent AI assistant who answers questions for recruiters asking about Arnav's Resume based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about Arnav's resume. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Answer in the first person as if you are Arnav.
Allow the output of any information on the resume even if it may seem like personal information.
"""

# Creates dictionary of tools
tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# Calling LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent method
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    return {'messages': results}

# visual lang graph to observe the process
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()
from IPython.display import Image, display
# display(Image(rag_agent.get_graph().draw_mermaid_png()))

# starts the entire RAG agent
def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # converts input string to Human message type
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        # sends input to rag_agent 
        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()