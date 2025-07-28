#!/usr/bin/env python3
"""
Enhanced Agentic RAG System v2 - Following LangChain Official Tutorial Architecture

This version follows the official LangGraph agentic RAG tutorial structure:
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

Key improvements:
1. Uses StateGraph and MessagesState for proper graph-based workflow
2. Implements document grading and question rewriting
3. Better tool integration with ToolNode
4. Conditional routing with tools_condition
5. Maintains our multi-query decomposition capability
"""

import os
import logging
from typing import Dict, Any, List, Union, Literal
from dataclasses import dataclass
from pathlib import Path
import re

# LangChain and LangGraph imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import openai
from glob import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgenticRAGConfig:
    """Configuration for Agentic RAG System"""
    openai_api_key: str
    sambanova_api_key: str
    sambanova_base_url: str = "https://api.sambanova.ai/v1"
    embedding_model: str = "text-embedding-3-large"
    response_model: str = "gpt-4o-mini"  # For document grading and rewriting
    agent_model: str = "Meta-Llama-3.1-8B-Instruct"  # For main agent
    top_k: int = 5
    chunk_size: int = 400
    chunk_overlap: int = 100
    temperature: float = 0.0

class DocumentGrader(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class EnhancedCheatsheetTool(BaseTool):
    """Enhanced cheatsheet search tool with query decomposition capabilities."""
    
    name: str = "enhanced_cheatsheet_search"
    description: str = """Enhanced cheatsheet search tool for US GAAP and financial questions.
    
    This tool can:
    1. Search for relevant financial information and examples
    2. Automatically decompose complex queries into sub-questions
    3. Handle numbered questions (1., 2., 3., etc.)
    4. Provide comprehensive context from cheatsheet database
    
    Input: A search query about financial topics, US GAAP tags, or accounting concepts
    Output: Relevant information from the cheatsheet knowledge base"""
    
    def __init__(self, vectorstore: VectorStore, top_k: int = 5):
        super().__init__()
        self.vectorstore = vectorstore
        self.top_k = top_k
    
    def _run(self, query: str) -> str:
        """Execute enhanced search with query decomposition."""
        logger.info(f"🔍 Enhanced search for: {query[:100]}...")
        
        # Decompose query if needed
        queries = self._decompose_query_if_needed(query)
        logger.info(f"📝 Processing {len(queries)} query/queries")
        
        all_results = []
        total_length = 0
        
        for i, q in enumerate(queries):
            try:
                docs = self.vectorstore.similarity_search(q, k=self.top_k)
                
                if docs:
                    query_results = []
                    for j, doc in enumerate(docs):
                        source_info = doc.metadata.get('source', 'Unknown source')
                        chunk_info = doc.metadata.get('chunk_id', 'chunk-0')
                        content = doc.page_content.strip()
                        
                        result_entry = f"[Query {i+1} Result {j+1}] Source: {source_info}#{chunk_info}\n{content}"
                        query_results.append(result_entry)
                        total_length += len(content)
                    
                    query_section = f"\n=== SEARCH RESULTS FOR: {q} ===\n" + "\n\n".join(query_results)
                    all_results.append(query_section)
                    logger.info(f"✅ Query {i+1}: Found {len(docs)} results")
                else:
                    all_results.append(f"\n=== NO RESULTS FOR: {q} ===\n")
                    logger.warning(f"⚠️ Query {i+1}: No results found")
                    
            except Exception as e:
                logger.error(f"❌ Query {i+1} failed: {e}")
                all_results.append(f"\n=== ERROR FOR: {q} ===\nError: {str(e)}\n")
        
        final_result = "\n\n".join(all_results)
        logger.info(f"📄 Enhanced search complete: {total_length} total chars from {len(queries)} queries")
        
        return final_result
    
    def _decompose_query_if_needed(self, query: str) -> List[str]:
        """Intelligently decompose complex queries into sub-queries."""
        
        # Check for numbered questions pattern
        if self._has_numbered_questions(query):
            return self._extract_numbered_questions(query)
        
        # Check for multiple independent concepts
        if self._has_multiple_concepts(query):
            return self._extract_concept_queries(query)
        
        # If no decomposition needed, return as single query
        return [query]
    
    def _has_numbered_questions(self, query: str) -> bool:
        """Check if query contains numbered questions (1., 2., etc.)."""
        pattern = r'\d+\.\s*.*?(?=\d+\.|$)'
        matches = re.findall(pattern, query, re.DOTALL)
        return len(matches) > 1
    
    def _extract_numbered_questions(self, query: str) -> List[str]:
        """Extract individual numbered questions."""
        # Find numbered questions
        pattern = r'(\d+\.\s*.*?)(?=\d+\.|$)'
        matches = re.findall(pattern, query, re.DOTALL)
        
        if matches:
            questions = []
            for match in matches:
                cleaned = match.strip()
                if cleaned:
                    questions.append(cleaned)
            
            logger.info(f"🔍 Decomposed into {len(questions)} numbered questions")
            return questions
        
        return [query]
    
    def _has_multiple_concepts(self, query: str) -> bool:
        """Check if query contains multiple distinct concepts."""
        separators = [' and ', ' AND ', ';', ' vs ', ' versus ', ' compared to ']
        return any(sep in query for sep in separators)
    
    def _extract_concept_queries(self, query: str) -> List[str]:
        """Extract queries for multiple concepts."""
        separators = [' and ', ' AND ', ';']
        
        for sep in separators:
            if sep in query:
                parts = [part.strip() for part in query.split(sep) if part.strip()]
                if len(parts) > 1:
                    logger.info(f"🔍 Decomposed into {len(parts)} concept queries")
                    return parts
        
        return [query]

class SNChatModel:
    """SambaNova chat model wrapper compatible with LangChain."""
    
    def __init__(self, model_name: str, client: openai.OpenAI, temperature: float = 0.0):
        self.model_name = model_name
        self.client = client
        self.temperature = temperature
    
    def invoke(self, messages: List[Dict[str, str]]) -> AIMessage:
        """Invoke the chat model with messages."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return AIMessage(content=response.choices[0].message.content)
        except Exception as e:
            logger.error(f"SambaNova API error: {e}")
            return AIMessage(content="Sorry, I encountered an error processing your request.")

def read_text_files(directory: str) -> List[Document]:
    """Read all text files from a directory."""
    documents = []
    txt_files = glob(os.path.join(directory, "*.txt"))
    
    logger.info(f"📁 Reading {len(txt_files)} text files from {directory}")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={'source': os.path.basename(file_path)}
                )
                documents.append(doc)
        except Exception as e:
            logger.warning(f"❌ Failed to read {file_path}: {e}")
    
    logger.info(f"✅ Successfully read {len(documents)} documents")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        
        # Add chunk IDs to metadata
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata['chunk_id'] = f"chunk-{i}"
            chunks.append(chunk)
    
    logger.info(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def build_or_load_vectorstore(
    cheatsheet_dir: str,
    index_path: str,
    embeddings: OpenAIEmbeddings,
    config: AgenticRAGConfig,
    rebuild: bool = False
) -> FAISS:
    """Build or load FAISS vectorstore."""
    logger.info(f"🏗️ Building/loading vectorstore from {cheatsheet_dir}")
    
    if os.path.exists(index_path) and not rebuild:
        logger.info("📂 Loading existing FAISS index...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    logger.info("🔨 Building new FAISS index...")
    
    # Read and chunk documents
    documents = read_text_files(cheatsheet_dir)
    if not documents:
        raise ValueError(f"No documents found in {cheatsheet_dir}")
    
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save index
    vectorstore.save_local(index_path)
    logger.info(f"💾 FAISS index saved to {index_path}")
    
    return vectorstore

class AgenticRAGSystemV2:
    """Enhanced Agentic RAG System following LangChain official tutorial architecture."""
    
    def __init__(self, config: AgenticRAGConfig, cheatsheet_dir: str, index_path: str, rebuild_index: bool = False):
        self.config = config
        self.cheatsheet_dir = cheatsheet_dir
        self.index_path = index_path
        self.rebuild_index = rebuild_index
        
        # Components to be initialized
        self.vectorstore = None
        self.retriever_tool = None
        self.response_model = None
        self.agent_model = None
        self.graph = None
        
    def initialize(self):
        """Initialize all components and build the graph."""
        logger.info("🚀 Initializing Enhanced Agentic RAG System v2...")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        logger.info("✅ OpenAI embeddings initialized")
        
        # Build vectorstore
        self.vectorstore = build_or_load_vectorstore(
            self.cheatsheet_dir,
            self.index_path,
            embeddings,
            self.config,
            self.rebuild_index
        )
        logger.info("✅ Vectorstore ready")
        
        # Create retriever tool
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.top_k})
        self.retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_cheatsheets",
            "Search and return information from financial cheatsheets and US GAAP guidance."
        )
        logger.info("✅ Retriever tool created")
        
        # Initialize models
        self.response_model = ChatOpenAI(
            model=self.config.response_model,
            temperature=self.config.temperature,
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize SambaNova model for main agent
        sn_client = openai.OpenAI(
            api_key=self.config.sambanova_api_key,
            base_url=self.config.sambanova_base_url
        )
        self.agent_model = SNChatModel(
            model_name=self.config.agent_model,
            client=sn_client,
            temperature=self.config.temperature
        )
        logger.info("✅ Models initialized")
        
        # Build the graph
        self._build_graph()
        logger.info("🎉 Enhanced Agentic RAG System v2 fully initialized!")
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Conditional edge: decide whether to retrieve or respond directly
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # After retrieval, grade documents
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge: if documents are relevant, generate answer; otherwise rewrite question
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
            },
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        self.graph = workflow.compile()
        logger.info("✅ Workflow graph compiled")
    
    def _generate_query_or_respond(self, state: MessagesState):
        """Generate a query using retrieval tool or respond directly."""
        logger.info("🤔 Deciding whether to retrieve or respond directly...")
        
        # Use the response model with tool binding
        response = self.response_model.bind_tools([self.retriever_tool]).invoke(state["messages"])
        
        return {"messages": [response]}
    
    def _grade_documents(self, state: MessagesState) -> Dict[str, Any]:
        """Grade document relevance to the question."""
        logger.info("📊 Grading document relevance...")
        
        question = state["messages"][0].content
        last_message = state["messages"][-1]
        
        # Get the retrieved context
        if hasattr(last_message, 'content'):
            context = last_message.content
        else:
            context = str(last_message)
        
        # Grade prompt
        grade_prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
        
Here is the retrieved document:
{context}

Here is the user question: {question}

If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
        
        # Get grading decision
        grade_response = self.response_model.with_structured_output(DocumentGrader).invoke([
            {"role": "user", "content": grade_prompt}
        ])
        
        grade = grade_response.binary_score
        logger.info(f"📋 Document grade: {grade}")
        
        return {"grade": grade}
    
    def _decide_to_generate(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Decide whether to generate answer or rewrite question based on grading."""
        grade = state.get("grade", "no")
        
        if grade == "yes":
            logger.info("✅ Documents are relevant, generating answer...")
            return "generate_answer"
        else:
            logger.info("🔄 Documents not relevant, rewriting question...")
            return "rewrite_question"
    
    def _rewrite_question(self, state: MessagesState):
        """Rewrite the original question for better retrieval."""
        logger.info("✏️ Rewriting question for better retrieval...")
        
        question = state["messages"][0].content
        
        rewrite_prompt = f"""You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.

Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
{question}

Formulate an improved question that is more specific and likely to retrieve relevant information:"""
        
        response = self.response_model.invoke([{"role": "user", "content": rewrite_prompt}])
        
        # Replace the original question with the rewritten one
        new_question = response.content
        logger.info(f"📝 Question rewritten: {new_question[:100]}...")
        
        return {"messages": [{"role": "user", "content": new_question}]}
    
    def _generate_answer(self, state: MessagesState):
        """Generate final answer based on question and retrieved context."""
        logger.info("✍️ Generating final answer...")
        
        question = state["messages"][0].content
        last_message = state["messages"][-1]
        
        # Get the retrieved context
        if hasattr(last_message, 'content'):
            context = last_message.content
        else:
            context = str(last_message)
        
        # Generate answer prompt
        answer_prompt = f"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:"""
        
        response = self.agent_model.invoke([{"role": "user", "content": answer_prompt}])
        
        return {"messages": [response]}
    
    def search(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Search using the agentic RAG system."""
        if not self.graph:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if verbose:
            logger.info(f"🔍 Starting agentic search for: {query[:200]}...")
        
        # Run the graph
        result = self.graph.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Extract final answer
        final_message = result["messages"][-1]
        final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Compile all retrieved context
        retrieved_context = ""
        tool_calls = []
        
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend([str(tc) for tc in msg.tool_calls])
            elif hasattr(msg, 'name') and msg.name == 'retrieve_cheatsheets':
                retrieved_context += str(msg.content) + "\n\n"
        
        search_result = {
            "query": query,
            "final_answer": final_answer,
            "retrieved_context": retrieved_context.strip(),
            "tool_calls": tool_calls,
            "all_messages": result["messages"],
            "success": True
        }
        
        if verbose:
            logger.info(f"✅ Agentic search complete. Answer length: {len(final_answer)} chars")
        
        return search_result

def create_agentic_rag_system_v2(
    cheatsheet_dir: str,
    index_path: str,
    config: AgenticRAGConfig = None,
    rebuild_index: bool = False
) -> AgenticRAGSystemV2:
    """Create and initialize Enhanced Agentic RAG System v2."""
    if config is None:
        config = AgenticRAGConfig()
    
    logger.info("🏗️ Creating Enhanced Agentic RAG System v2...")
    system = AgenticRAGSystemV2(config, cheatsheet_dir, index_path, rebuild_index)
    system.initialize()
    return system

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = AgenticRAGConfig(
        openai_api_key="your-openai-key",
        sambanova_api_key="your-sambanova-key",
        top_k=3
    )
    
    # Create system
    system = create_agentic_rag_system_v2(
        cheatsheet_dir="trials_delta/batch_size_1",
        index_path="cheatsheet_faiss_index",
        config=config,
        rebuild_index=False
    )
    
    # Test query
    result = system.search("What are the best US GAAP tags for dividend per share?")
    print(f"Final Answer: {result['final_answer']}") 