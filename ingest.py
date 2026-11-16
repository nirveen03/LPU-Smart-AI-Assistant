from rag import ingest_pdfs

if __name__ == "__main__":
    try:
        count = ingest_pdfs(replace_existing=True)
        print(f"\nIngestion complete. Total indexed chunks: {count}")
        print("You can now run: streamlit run app.py")
    except Exception as exc:
        print(f"\nIngestion failed: {exc}")
        raise
