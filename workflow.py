from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
from config import GOOGLE_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    max_tokens=500,
    temperature=0.1
)

def create_workflow(retriever):
    """Create LangGraph workflow for multi-agent system"""
    # Retriever Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "legal_document_retriever",
        "Retrieve relevant legal information from documents"
    )

    # Define Agent State
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # Create Workflow Graph
    workflow = StateGraph(AgentState)

    # Define Nodes
    def query_agent(state: AgentState):
        """Query Agent: Retrieve relevant legal information"""
        messages = state['messages']
        last_message = messages[-1]

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(last_message.content)

        return {
            "messages": [
                HumanMessage(content=f"Retrieved Documents: {retrieved_docs}")
            ]
        }

    def summarization_agent(state: AgentState):
        """Summarization Agent: Simplify legal concepts"""
        messages = state['messages']
        last_message = messages[-1]

        # Create summarization prompt
        summarization_prompt = PromptTemplate(
            template="""You are a legal expert assistant. Simplify the following legal information into clear, concise language:

            Retrieved Legal Context: {context}

            Original Query: {query}

            Provide a clear, simplified explanation that a layperson can understand. Focus on key points and practical implications. If the retriever doesn't provide any information or if the query is not related to law, simply state that the question is not law-related and summarize it in easy language.""",
            input_variables=["context", "query"]
        )

        # Summarization chain
        summarization_chain = (
            summarization_prompt
            | llm
            | StrOutputParser()
        )

        # Prepare context (first message is retrieved documents)
        context = messages[0].content
        query = messages[-1].content

        # Generate summary
        summary = summarization_chain.invoke({
            "context": context,
            "query": query
        })

        return {
            "messages": [HumanMessage(content=summary)]
        }

    # Add nodes to workflow
    workflow.add_node("query_agent", query_agent)
    workflow.add_node("summarization_agent", summarization_agent)

    # Define edges
    workflow.add_edge(START, "query_agent")
    workflow.add_edge("query_agent", "summarization_agent")
    workflow.add_edge("summarization_agent", END)

    # Compile workflow
    app = workflow.compile()
    return app
