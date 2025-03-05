import streamlit as st
import logging
from document_loader import load_documents
from vector_store import vectordb
from workflow import create_workflow
from langchain_core.messages import HumanMessage
from memory import SessionManager  # Import the SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state variables
if 'retriever' not in st.session_state:
    try:
        logger.info("Creating vector store...")
        st.session_state.retriever = vectordb()
        logger.info("Creating workflow...")
        st.session_state.app = create_workflow(st.session_state.retriever)
        logger.info("Initialization complete.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        st.error("Initialization failed. Please check the logs for more details.")

# Initialize the SessionManager in the session state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()

def process_query(query, session_id):
    """Process user query through multi-agent system"""
    try:
        logger.info(f"Processing query: {query}")
        # Get the session history
        session_history = st.session_state.session_manager.get_session_history(session_id)
        # Add the query to the session history
        session_history.add_message(HumanMessage(content=query))
        # Process the query
        result = st.session_state.app.invoke({"messages": session_history.messages})
        # Add the response to the session history
        session_history.add_message(result['messages'][-1])
        logger.info(f"Query processed successfully. Response: {result['messages'][-1].content}")
        return result['messages'][-1].content
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        st.error("Query processing failed. Please check the logs for more details.")
        return None

# Streamlit UI
st.title("Legal Assistant Bot")
st.write("Ask your legal questions and get simplified explanations.")

# Input box for user query
query = st.text_input("Enter your query:")
session_id = st.text_input("Enter your session ID:", "default_session")

if st.button("Submit"):
    if query:
        response = process_query(query, session_id)
        if response:
            st.write(f"**Response:** {response}")
    else:
        st.write("Please enter a query.")

# Display conversation history
st.write("### Conversation History")
session_history = st.session_state.session_manager.get_session_history(session_id)
for message in session_history.messages:
    st.write(f"**{message.type}:** {message.content}")
