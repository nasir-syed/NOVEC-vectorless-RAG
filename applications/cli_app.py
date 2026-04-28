import os
import sys
import logging
from typing import List, Dict, Any

from novec import RAGConfig, PageIndexAPI, RAGEngine, QueryExecutor, setup_logger

from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.rule import Rule
from rich.table import Table

# SETUP & CONFIGURATION

# logger
logger = setup_logger(
    name=__name__,
    level=logging.INFO
)

try:
    config = RAGConfig(logger=logger)
except ValueError as e:
    logger.error(str(e))
    print(f"Error: {str(e)}")
    sys.exit(1)

_theme = Theme({
    "success": "bold green",
    "error":   "bold red",
    "info":    "cyan",
    "header":  "bold white",
})

console = Console(theme=_theme)


# UTILITY FUNCTIONS (for CLI-specific formatting)

def print_header(text: str) -> None:
    console.print()
    console.print(Panel(f"[header]{text}[/header]", expand=False))
    console.print()

def print_success(text: str) -> None:
    console.print(f"[success]SUCCESS[/success]  {text}")

def print_error(text: str) -> None:
    console.print(f"[error]ERROR[/error]    {text}")

def print_info(text: str) -> None:
    console.print(f"[info]INFO[/info]     {text}")

def print_documents(documents: list) -> None:
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("#",           style="dim",        width=4)
    table.add_column("Name",        style="bold white")
    table.add_column("Description", style="white")

    for idx, doc in enumerate(documents, 1):
        table.add_row(
            str(idx),
            doc.get("name", "Unknown"),
            doc.get("description", "No description"),
        )

    console.print(table)

def print_separator() -> None:
    console.print(Rule(style="dim"))

def clear_screen() -> None:
    console.clear()

# MENU OPERATIONS

class MenuHandler:

    def __init__(self):
        self.api = PageIndexAPI(config)
        self.rag_engine = RAGEngine(config)
        self.query_executor = QueryExecutor(self.api, self.rag_engine)
        logger.info("Menu handler initialized")

    def display_main_menu(self) -> str:
        print_header("NOVEC RAG (CLI)")
        print("Choose an option:")
        print("  1. Write a query")
        print("  2. Upload a document")
        print("  3. Delete a document")
        print("  4. Exit")
        print()
        
        while True:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                return choice
            print_error("Invalid choice. Please enter 1-4.")

    def upload_document(self) -> None:
        print_header("Upload a Document")

        while True:
            file_path = input("Enter the file path: ").strip()

            if not file_path:
                print_error("File path cannot be empty")
                continue

            file_path = os.path.expanduser(file_path)

            if not os.path.exists(file_path):
                print_error(f"File not found: {file_path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != "y":
                    break
                continue

            file_name = os.path.basename(file_path)
            print_info(f"Uploading {file_name}...")
            doc_id = self.api.upload_document(file_path)
            
            if doc_id:
                print_success(f"Document uploaded! ID: {doc_id}")
                if self.api.wait_for_indexing(doc_id):
                    print_success("Document is ready to use!")
                    logger.info(f"Document {doc_id} uploaded and indexed successfully")
                else:
                    print_error("Document indexing failed")
                    logger.error(f"Indexing failed for doc_id {doc_id}")
            break

        input("\nPress Enter to return to menu...")

    def delete_document(self) -> None:
        print_header("Delete a Document")

        documents = self.api.fetch_documents()
        if not documents:
            print_error("No documents available or failed to fetch documents")
            logger.warning("Delete operation: no documents available")
            input("\nPress Enter to return to menu...")
            return

        print(f"Available documents ({len(documents)}):\n")
        print_documents(documents)

        while True:
            selection = input("Enter document number to delete (or 'c' to cancel): ").strip()

            if selection.lower() == "c":
                print_info("Deletion cancelled")
                break

            try:
                index = int(selection) - 1
                if 0 <= index < len(documents):
                    selected_doc = documents[index]
                    doc_id = selected_doc.get("id")
                    doc_name = selected_doc.get("name", "Unknown")
                    logger.info(f"Selected document: id={doc_id}, name={doc_name}")

                    confirm = input(f"Delete '{doc_name}'? (y/n): ").strip().lower()
                    if confirm == "y":
                        if self.api.delete_document(doc_id):
                            logger.info(f"Document {doc_id} deleted successfully")
                        else:
                            print_error("Failed to delete document")
                            logger.error(f"Failed to delete doc_id {doc_id}")
                    else:
                        print_info("Deletion cancelled")
                    break
                else:
                    print_error(f"Invalid selection. Please enter 1-{len(documents)}")
            except ValueError:
                print_error("Invalid input. Please enter a number or 'c'")

        input("\nPress Enter to return to menu...")

    def write_query(self) -> None:
        print_header("Write a Query")

        documents = self.api.fetch_documents()
        if not documents:
            print_error("No documents available for querying")
            logger.warning("Query operation: no documents available")
            input("\nPress Enter to return to menu...")
            return

        print(f"Available documents ({len(documents)}):\n")
        print_documents(documents)

        selected_docs = self._get_document_selection(documents)
        if not selected_docs:
            print_info("No documents selected")
            input("\nPress Enter to return to menu...")
            return

        print()
        query = input("Enter your query: ").strip()
        if not query:
            print_error("Query cannot be empty")
            input("\nPress Enter to return to menu...")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing query: {query}")
        logger.info(f"Documents selected: {[d.get('name') for d in selected_docs]}")
        logger.info(f"{'='*60}")

        def progress_callback(step: str, message: str):
            if step == "STEP_1":
                print_info(message)
            elif step == "STEP_2":
                print_info(message)
            elif step == "STEP_3":
                print_info(message)
            elif step == "ERROR":
                print_error(message)

        print_separator()
        result = self.query_executor.execute_query(query, selected_docs, progress_callback)

        if result["success"]:
            print_separator()
            print("Answer:")
            print(result["answer"])
            print_separator()
            
            answer = result["answer"]
            if '[' in answer and ']' in answer:
                print_success("Answer includes proper citations")
                logger.info("Answer validation: Citations detected")
            else:
                print_info("Answer may need additional citations")
                logger.warning("Answer validation: Limited citations detected")
            
            logger.info("Query processing completed successfully")
        else:
            print_error(f"Query failed: {result['error']}")
            logger.error(f"Query execution failed: {result['error']}")

        input("\nPress Enter to return to menu...")

    def _get_document_selection(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Select documents (up to 3) by entering their numbers separated by commas.")
        print("Example: 1,2,3\n")

        while True:
            selection = input("Enter document numbers (or 'c' to cancel): ").strip()

            if selection.lower() == "c":
                return []

            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]

                if len(indices) > 3:
                    print_error("You can select up to 3 documents")
                    continue

                if len(indices) != len(set(indices)):
                    print_error("Duplicate selections detected")
                    continue

                if not all(0 <= idx < len(documents) for idx in indices):
                    print_error(f"Invalid selection. Please enter numbers 1-{len(documents)}")
                    continue

                selected_docs = [documents[idx] for idx in indices]
                logger.info(f"Selected {len(selected_docs)} document(s)")
                for doc in selected_docs:
                    logger.info(f"  - {doc.get('name')} (id={doc.get('id')})")
                print_success(f"Selected {len(selected_docs)} document(s)")
                return selected_docs

            except ValueError:
                print_error("Invalid input. Please enter comma-separated numbers")

    def run(self) -> None:
        logger.info("Starting CLI application")
        
        while True:
            choice = self.display_main_menu()

            if choice == "1":
                self.write_query()
            elif choice == "2":
                self.upload_document()
            elif choice == "3":
                self.delete_document()
            elif choice == "4":
                print_info("Exiting application. Goodbye!")
                logger.info("Application exited normally")
                sys.exit(0)

# MAIN ENTRY POINT

def main():
    try:
        menu_handler = MenuHandler()
        menu_handler.run()
    except KeyboardInterrupt:
        print_info("\nApplication interrupted by user")
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
