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
    model="gpt-4o-mini", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# path to the resume 
pdf_path = "/Users/aps/Projects/Chat-With-my-Resume/RAG Resume.pdf"


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

# Optimized chunking for resume content
# Resume content benefits from larger chunks to maintain context
# and smaller overlap to avoid redundancy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increased for better context retention
    chunk_overlap=50,  # Reduced to minimize redundancy
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for resume structure
)

# applies the chunking settings and add metadata for better filtering
pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

# Add metadata to each chunk for better filtering and context
for i, doc in enumerate(pages_split):
    doc.metadata.update({
        'chunk_id': i,
        'source': 'arnav_resume',
        'page_number': doc.metadata.get('page', 0) + 1
    })

persist_directory = r"/Users/aps/Projects/Chat-With-my-Resume/VectorDB"
collection_name = "arnav_resume"

# Function to recreate vector database with optimized settings
def recreate_vector_db():
    """Recreate the vector database with new optimized settings."""
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("Removed old vector database to apply new settings...")
    
    os.makedirs(persist_directory)
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created optimized ChromaDB vector store with {len(pages_split)} chunks!")
    return vectorstore

# Check if we need to recreate the database (force recreation for new settings)
force_recreate = True  # Set to True to apply new chunking and retrieval settings

if force_recreate or not os.path.exists(persist_directory):
    vectorstore = recreate_vector_db()
else:
    print(f"Loading existing ChromaDB vector store...")
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        print(f"Loaded existing ChromaDB vector store!")
    except Exception as e:
        print(f"Error loading existing database: {e}")
        print("Recreating with new settings...")
        vectorstore = recreate_vector_db()

# Optimized retriever for resume content
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={
        'k': 3  # Reduced for more focused results
    }
)

# @ tool makes it callable by LLM
@tool
# Enhanced retriever tool with better context and filtering
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns relevant information from Arnav's resume document.
    Use this to find specific details about education, skills, projects, experience, or any other resume content.
    """
    # sends query to retriever to get top k outputs that answer the prompt from vector DB
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in my resume for that query. Please try rephrasing your question or ask about a different topic."
    
    results = []
    # Enhanced output with better formatting and context
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page_number', 'Unknown')
        results.append(f"Relevant Information (Page {page_num}):\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

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

# Enhanced system prompt optimized for resume content
system_prompt = """
You are Arnav Sharma, a Computer Science student at Penn State University. You are speaking directly to recruiters and hiring managers about your background, skills, and experience.

Your role:
- Answer questions about your resume, education, projects, skills, and experience
- Provide specific details about your technical skills, projects, and achievements
- Be professional, confident, and enthusiastic about opportunities
- Use the retriever tool to find specific information from your resume when needed
- If you need to look up information, do so before responding
- Always speak in first person as Arnav

Key areas you can discuss:
- Education: Penn State University, Computer Science
- Technical Skills: Programming languages, frameworks, tools
- Projects: Personal and academic projects with specific details
- Experience: Internships, work experience, leadership roles
- Contact Information: Available when relevant
- Career Goals: Your passion for AI/ML and full-stack development

Be specific and detailed in your responses. If asked about something not in your resume, politely mention that it's not covered in your current resume but you'd be happy to discuss it further.
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