from src.rag_pipeline import RAGPipeline

pdf_path = "data/Employee_Handbook.pdf"

#Crate the pipeline
pipeline=RAGPipeline()

print("Building the vector database...")
pipeline.build_database(pdf_path)
print("Database ready for queries")

while True:
    question=input("\nASK a question( or type 'exit'): ")

    if question.lower()=="exit":
        print("Goodbye!")
        break
    answer,sources=pipeline.ask(question)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    for source in sources:
        print(source.metadata)