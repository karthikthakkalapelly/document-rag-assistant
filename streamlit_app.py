import os
import streamlit as st
from src.rag_pipeline import RAGPipeline
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document RAG Assistant")
st.write("Ask questions about your uploaded PDFs")

#Initialize pipeline

if "pipeline" not in st.session_state:
    st.session_state.pipeline=RAGPipeline()

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

if "database_ready" not in st.session_state:
    st.session_state.database_ready=False

if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names=[]

#Side bar
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files=st.file_uploader("Choose PDF files",type=["pdf"],accept_multiple_files=True)

    if uploaded_files :
        uploaded_names=[file.name for file in uploaded_files]

        #Build database only if a different pdf is uploaded
        if uploaded_names != st.session_state.uploaded_pdf_names:

            os.makedirs("data",exist_ok=True)
            pdf_paths=[]
            for uploaded_file in uploaded_files:
                pdf_path=os.path.join("data",
                                  uploaded_file.name)
            
                with open(pdf_path,"wb")as file:
                    file.write(uploaded_file.getbuffer())
                pdf_paths.append(pdf_path)

            
            #Create a  new pipeline for the uploaded PDF
            st.session_state.pipeline=RAGPipeline()

            with st.spinner("Building vector database..."):
                st.session_state.pipeline.build_database(pdf_paths)
            if st.session_state.pipeline.ocr_documents:
                    st.warning(
        "📄 OCR was used for:\n\n" +
        "\n".join(st.session_state.pipeline.ocr_documents)
    )
            else:
                    st.success("✅ Text-based PDFs detected. OCR was not required.")
                
            st.session_state.database_ready=True
            st.session_state.uploaded_pdf_names=uploaded_names
            st.session_state.messages=[]
            st.success("Database Created Successfully!")

    st.divider()

    #Document Information
    if st.session_state.database_ready:
        st.subheader("📄 Uploaded Documents")
        st.write(f"Total PDFs:{len(st.session_state.pipeline.pdf_names)}")
        st.write(f"Total pages:{st.session_state.pipeline.total_pages}")
        st.write(f"Total chunks:{st.session_state.pipeline.total_chunks}")
        st.write("###Files")
        
        for pdf in st.session_state.pipeline.pdf_names:
            st.success(pdf)

    
    st.divider()    
#Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages=[]
        st.rerun()

#Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Chat input
question=st.chat_input("Ask your questtion")

#Process user question

if question:
    if not st.session_state.database_ready:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    #Show user message
    st.session_state.messages.append({
        "role":"user",
        "content":question
    })
    with st.chat_message("user"):
        st.markdown(question)

#Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
         answer,sources,confidence=st.session_state.pipeline.ask(question)
        #Remove duplicate page numbers
         

# ---------------------------------------
# Group source pages by document
# ---------------------------------------

         document_sources = defaultdict(set)
         for source in sources:
            pdf_name = os.path.basename(
             source.metadata["source"]
    )

            document_sources[pdf_name].add(
            source.metadata["page"]
    )

# ---------------------------------------
# Build source text
# ---------------------------------------

         source_text = ""
         for pdf_name, pages in document_sources.items():
            page_list = ", ".join(
              map(str, sorted(pages))
    )

            source_text += (
              f"• {pdf_name} → Page(s): {page_list}\n"
    )

            response = f"""
            {answer}

            ---
            📊 **Confidence:** {confidence}%
            📚 **Sources**
            {source_text}
            """
         st.markdown(response)
         with st.expander("🔍 View Retrieved Context"):
             for index, source in enumerate(sources,start=1):
                st.markdown(
                f"### Result {index}"
        )

                st.write(
                f"**Page:** {source.metadata['page']}"
        )

                st.write(source.page_content)

                st.divider()

    #Save assistant response
    st.session_state.messages.append({
        "role":"assistant",
        "content":response
    })