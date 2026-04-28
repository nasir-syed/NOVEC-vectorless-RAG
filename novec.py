import os
import json
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import requests
from openai import OpenAI
from config import Config

# CONFIGURATION & SETUP

# central configuration for the RAG system
class RAGConfig:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_url: str = "https://api.pageindex.ai",
        timeout: int = 10,
        logger: Optional[logging.Logger] = None
    ):
       
        self.api_key = api_key or Config.PAGEINDEX_API_KEY
        self.openai_key = openai_key or Config.OPENAI_BASE_KEY
        self.model_name = model_name or Config.MODEL_NAME
        self.api_url = api_url
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("PAGEINDEX_API_KEY not configured")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not configured")


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
   
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.addHandler(console_handler)
    return logger

# PAGEINDEX API OPERATIONS

# Handles all PageIndex API interactions
class PageIndexAPI:
    
    
    def __init__(self, config: RAGConfig):

        self.config = config
        self.api_key = config.api_key
        self.base_url = config.api_url
        self.headers = {"api_key": self.api_key}
        self.timeout = config.timeout
        self.logger = config.logger
        self.logger.info("PageIndex API client initialized")

    def upload_document(self, file_path: str) -> Optional[str]:
       
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None

            if not os.path.isfile(file_path):
                self.logger.error(f"Path is not a file: {file_path}")
                return None

            if not os.access(file_path, os.R_OK):
                self.logger.error(f"File is not readable: {file_path}")
                return None

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"File is empty: {file_path}")
                return None

            file_name = os.path.basename(file_path)
            self.logger.info(f"Uploading {file_name} ({file_size / 1024:.2f} KB)...")
            self.logger.info(f"POST request to: {self.base_url}/doc/")

            with open(file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"{self.base_url}/doc/",
                    headers=self.headers,
                    files=files,
                    timeout=self.timeout
                )

            self.logger.info(f"Upload response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                doc_id = data.get("doc_id")
                self.logger.info(f"Document uploaded successfully. doc_id: {doc_id}")
                return doc_id
            else:
                self.logger.error(f"Upload failed: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error("Upload request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
            return None

    def wait_for_indexing(self, doc_id: str, timeout: int = 300, poll_interval: int = 5) -> bool:
        try:
            start_time = time.time()
            self.logger.info("Waiting for document indexing to complete...")

            while time.time() - start_time < timeout:
                response = requests.get(
                    f"{self.base_url}/doc/{doc_id}",
                    headers=self.headers,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    self.logger.error(f"Status check failed: HTTP {response.status_code}")
                    return False

                data = response.json()
                status = data.get("status")
                self.logger.info(f"Document {doc_id} status: {status}")

                if status == "completed":
                    self.logger.info("Indexing completed successfully")
                    return True
                elif status == "failed":
                    self.logger.error(f"Document {doc_id} processing failed")
                    return False

                time.sleep(poll_interval)

            self.logger.error(f"Indexing timeout after {timeout} seconds")
            return False

        except requests.exceptions.Timeout:
            self.logger.error("Status check timed out")
            return False
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during indexing wait: {str(e)}", exc_info=True)
            return False

    def fetch_documents(self, limit: int = 50, offset: int = 0) -> Optional[List[Dict[str, Any]]]:
        try:
            self.logger.info(f"Fetching documents (limit={limit}, offset={offset})")
            
            response = requests.get(
                f"{self.base_url}/docs",
                headers=self.headers,
                params={"limit": limit, "offset": offset},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", []) if isinstance(data, dict) else data
                
                if not documents and "docs" in data:
                    documents = data.get("docs", [])
                
                self.logger.info(f"Fetched {len(documents)} documents")
                return documents
            else:
                self.logger.error(f"Fetch failed: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error("Fetch request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching documents: {str(e)}", exc_info=True)
            return None

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.logger.info(f"Deleting document: {doc_id}")
            
            response = requests.delete(
                f"{self.base_url}/doc/{doc_id}/",
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code in [200, 204]:
                self.logger.info(f"Document {doc_id} deleted successfully")
                return True
            else:
                self.logger.error(f"Delete failed: HTTP {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            self.logger.error("Delete request timed out")
            return False
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during delete: {str(e)}", exc_info=True)
            return False

    def get_tree(self, doc_id: str, node_summary: bool = True) -> Optional[List[Dict[str, Any]]]:
        try:
            self.logger.info(f"Fetching tree for document: {doc_id}")
            
            response = requests.get(
                f"{self.base_url}/tree/{doc_id}",
                headers=self.headers,
                params={"node_summary": node_summary},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                tree = data.get("result", []) if isinstance(data, dict) else data
                self.logger.info(f"Tree fetched for {doc_id}")
                return tree
            else:
                self.logger.error(f"Failed to fetch tree: HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            self.logger.error("Get tree request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching tree: {str(e)}", exc_info=True)
            return None

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            self.logger.info(f"Fetching metadata for document: {doc_id}")
            
            response = requests.get(
                f"{self.base_url}/doc/{doc_id}",
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                metadata = response.json()
                status = metadata.get("status")
                page_count = metadata.get("pageNum", metadata.get("pages", 0))
                
                self.logger.info(f"Document {doc_id}: status={status}, pages={page_count}")
                
                return {
                    "status": status,
                    "page_count": page_count,
                    "name": metadata.get("name"),
                    "created_at": metadata.get("createdAt"),
                    "description": metadata.get("description")
                }
            else:
                self.logger.error(f"Failed to fetch metadata: HTTP {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching metadata: {str(e)}")
            return None

# RAG OPERATIONS
# Handles RAG-based retrieval and answer generation
class RAGEngine:

    def __init__(self, config: RAGConfig):

        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_key)
        self.model_name = config.model_name
        self.logger = config.logger
        self.logger.info("RAG engine initialized")

    def extract_document_structure(self, tree: List[Dict[str, Any]], max_depth: int = 2) -> str:
        
        def format_node(node: Dict, depth: int = 0) -> str:
            if depth > max_depth:
                return ""
            
            indent = "  " * depth
            line = f"{indent}- {node.get('title', 'Unknown')} (Page {node.get('page_index', '?')})"
            
            result = line
            if node.get("nodes") and depth < max_depth:
                for child in node.get("nodes", []):
                    child_text = format_node(child, depth + 1)
                    if child_text:
                        result += "\n" + child_text
            
            return result
        
        structure_lines = []
        for node in tree:
            node_text = format_node(node)
            if node_text:
                structure_lines.append(node_text)
        
        structure = "\n".join(structure_lines)
        self.logger.info(f"Extracted document structure ({len(structure)} chars)")
        return structure

    def identify_relevant_sections(
        self,
        query: str,
        document_structure: str,
        doc_name: str
    ) -> Optional[Dict[str, Any]]:
        
        try:
            self.logger.info(f"Identifying relevant sections in {doc_name}")
            
            prompt = f"""You are helping identify relevant sections in a document for answering a query.
            
            Document: {doc_name}
            Query: {query}

            Document Structure (Table of Contents):
            {document_structure}

            Based on the query, which sections of the document are most likely to contain relevant information?
            List the section titles and page ranges that should be examined.
            Be specific about which sections to read.

            Reply in this JSON format:
            {{
            "relevant_sections": [
                {{"title": "Section Title", "page": page_number, "relevance": "why this is relevant"}}
            ],
            "strategy": "brief explanation of your selection strategy"
            }}"""

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            sections = result.get('relevant_sections', [])
            
            self.logger.info(f"Identified {len(sections)} relevant sections")
            
            return result

        except Exception as e:
            self.logger.error(f"Error identifying sections: {str(e)}", exc_info=True)
            return None

    def extract_full_content(self, tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       
        all_nodes = []
        
        def traverse(nodes, parent_title=""):
            for node in nodes:
                node_with_context = node.copy()
                node_with_context["parent_section"] = parent_title
                all_nodes.append(node_with_context)
                
                if node.get("nodes"):
                    traverse(node.get("nodes"), node.get("title", parent_title))
        
        traverse(tree)
        self.logger.info(f"Extracted {len(all_nodes)} total content nodes")
        return all_nodes

    def extract_asset_references(self, nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
       
        assets = {
            "figures": [],
            "charts": [],
            "tables": [],
            "images": [],
            "diagrams": []
        }
        
        patterns = {
            "figures": r"(?:figure|fig\.?)[\s\-:]*\d+",
            "charts": r"(?:chart|graph|plot)[\s\-:]*\d+",
            "tables": r"(?:table|tab\.?)[\s\-:]*\d+",
            "images": r"(?:image|img|photo)[\s\-:]*\d+",
            "diagrams": r"(?:diagram|flowchart)[\s\-:]*\d+"
        }
        
        for node in nodes:
            text = node.get("text", "").lower()
            if not text:
                continue
            
            for asset_type, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match not in assets[asset_type]:
                        assets[asset_type].append(match)
        
        total_assets = sum(len(v) for v in assets.values())
        if total_assets > 0:
            self.logger.info(f"Found {total_assets} asset references")
        
        return assets

    def compress_tree(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        out = []
        for n in nodes:
            entry = {
                "node_id": n["node_id"],
                "title": n["title"],
                "page": n.get("page_index", "?"),
                "summary": n.get("text", "")[:250]
            }
            if n.get("nodes"):
                entry["children"] = self.compress_tree(n["nodes"])
            out.append(entry)
        return out

    def llm_tree_search(self, query: str, tree: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        
        try:
            self.logger.info(f"Starting LLM tree search for query: {query}")
            compressed_tree = self.compress_tree(tree)

            prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
            Your task: identify which node IDs most likely contain the answer to the query.
            Think step-by-step about which sections are relevant.

            Query: {query}

            Document Tree:
            {json.dumps(compressed_tree, indent=2)}

            Reply ONLY in this exact JSON format:
            {{
            "thinking": "<your step-by-step reasoning>",
            "node_list": ["node_id1", "node_id2"]
            }}"""

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            node_ids = result.get('node_list', [])
            
            self.logger.info(f"LLM tree search found {len(node_ids)} relevant nodes")
            
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in tree search: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error during tree search: {str(e)}", exc_info=True)
            return None

    def find_nodes_by_ids(self, tree: List[Dict[str, Any]], target_ids: List[str]) -> List[Dict[str, Any]]:
        
        found = []
        for node in tree:
            if node["node_id"] in target_ids:
                found.append(node)
                self.logger.debug(f"Found node: {node.get('title', 'Unknown')}")
            if node.get("nodes"):
                found.extend(self.find_nodes_by_ids(node["nodes"], target_ids))
        
        self.logger.info(f"Total nodes retrieved: {len(found)}")
        return found
    
    # JSON post-processing helpers

    _FALLBACK_JSON: Dict[str, Any] = {
        "response": "I'm unable to answer that based on the available information.",
        "citations": []
    }

    def _parse_llm_json(self, raw: str) -> Dict[str, Any]:
        def _normalise(obj: Any) -> Dict[str, Any]:
            if not isinstance(obj, dict):
                return dict(self._FALLBACK_JSON)

            response_val = obj.get("response", "")
            if not isinstance(response_val, str) or not response_val.strip():
                response_val = self._FALLBACK_JSON["response"]

            citations_val = obj.get("citations", [])
            if isinstance(citations_val, str):
                # Some models return citations as a newline-separated string
                citations_val = [c.strip() for c in citations_val.splitlines() if c.strip()]
            if not isinstance(citations_val, list):
                citations_val = []

            return {"response": response_val, "citations": citations_val}

        if not raw or not raw.strip():
            self.logger.warning("_parse_llm_json: received empty raw string")
            return dict(self._FALLBACK_JSON)

        # --- strategy 1: direct parse ---
        try:
            return _normalise(json.loads(raw))
        except json.JSONDecodeError:
            pass

        # --- strategy 2: strip markdown fences ---
        stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped.strip())
        try:
            return _normalise(json.loads(stripped))
        except json.JSONDecodeError:
            pass

        # --- strategy 3: extract first {...} block ---
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return _normalise(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        self.logger.error("_parse_llm_json: all parse strategies failed; using fallback")
        return dict(self._FALLBACK_JSON)

    def _build_citations_from_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        seen: set = set()
        citations: List[str] = []
        for node in nodes:
            title = node.get("title", "Unknown Section")
            page  = node.get("page_index", "?")
            ref   = f"{title} (Page {page})"
            if ref not in seen:
                seen.add(ref)
                citations.append(ref)
        return citations

    
    # Public generation methods

    def generate_answer(self, query: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        fallback = dict(self._FALLBACK_JSON)

        try:
            self.logger.info(f"Generating answer for query: {query}")

            # Guard: no nodes to build context from
            if not nodes:
                self.logger.warning("generate_answer: no nodes provided")
                return fallback

            # Build context
            context_parts: List[str] = []
            for node in nodes:
                title        = node.get("title", "Unknown")
                page         = node.get("page_index", "?")
                text_content = node.get("text", "") or "Content not available."
                context_parts.append(
                    f"[Section: '{title}' | Page {page}]\n{text_content}"
                )

            context = "\n\n---\n\n".join(context_parts)
            self.logger.info(f"Built context from {len(context_parts)} sections")

            prompt = f"""You are an expert document analyst.
            Answer the question using ONLY the provided context.
            For every claim you make, cite the section title and page number.

            You MUST respond with a single, valid JSON object — no text before or after it.
            The JSON must have exactly these two keys:
            "response"  : a clear, well-formatted natural-language answer (string)
            "citations" : an array of citation strings, e.g. ["Section Title (Page 3)", ...]
                            Use an empty array [] if no reliable sources are available.

            Strict JSON rules: no trailing commas, all strings properly escaped, no comments.

            Question: {query}

            Context:
            {context}

            JSON response:"""

            raw_response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            raw_text = raw_response.choices[0].message.content
            self.logger.debug(f"Raw LLM output (generate_answer): {raw_text[:300]}")

            result = self._parse_llm_json(raw_text)

            # If LLM returned an empty citations list but we have node metadata,
            # synthesise citations so callers always receive useful references.
            if not result["citations"]:
                self.logger.info(
                    "generate_answer: citations array empty; synthesising from node metadata"
                )
                result["citations"] = self._build_citations_from_nodes(nodes)

            self.logger.info(
                f"generate_answer: done — "
                f"{len(result['citations'])} citation(s)"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"generate_answer: unexpected error: {str(e)}", exc_info=True
            )
            return fallback

    def generate_cited_answer(
        self,
        query: str,
        context_by_doc: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        fallback = dict(self._FALLBACK_JSON)

        try:
            self.logger.info(f"Generating cited answer for query: {query}")
            self.logger.info(f"Processing {len(context_by_doc)} document(s)")
            
            if not context_by_doc:
                self.logger.warning("generate_cited_answer: context_by_doc is empty")
                return fallback

            context_parts: List[str] = []
            all_nodes_seen: List[Dict[str, Any]] = []

            for doc_name, doc_data in context_by_doc.items():
                nodes      = doc_data.get("nodes", []) or []
                strategy   = doc_data.get("strategy", "Full document extraction")
                page_count = doc_data.get("page_count", "?")

                self.logger.info(
                    f"Processing '{doc_name}' ({page_count} pages): {strategy}"
                )

                for node in nodes:
                    text_content = node.get("text", "") or ""
                    if not text_content.strip():
                        self.logger.debug(
                            f"Skipping node without text: {node.get('title', 'unknown')}"
                        )
                        continue

                    title = node.get("title", "Unknown")
                    page  = node.get("page_index", "?")

                    context_parts.append(
                        f"[{doc_name} | Page {page}]\n{title}\n---\n{text_content}"
                    )
                    
                    all_nodes_seen.append({**node, "_doc_name": doc_name})

            if not context_parts:
                self.logger.warning(
                    "generate_cited_answer: no text content found in any node"
                )
                return fallback

            context = "\n\n---\n\n".join(context_parts)
            self.logger.info(
                f"Built context from {len(context_parts)} section(s) across all documents"
            )


            prompt = f"""You are an expert analyst answering questions based on provided documents.

            INSTRUCTIONS:
            1. Answer the question using ONLY information from the provided documents.
            2. For EVERY factual claim, include an inline citation in the format: [Document Name, Page X]
            3. If documents contain conflicting information, acknowledge this explicitly.
            4. If the documents lack sufficient information, state this clearly in "response".
            5. Do NOT use external knowledge — stick strictly to the provided documents.

            You MUST respond with a single, valid JSON object — no text before or after it.
            The JSON must have exactly these two keys:
            "response"  : a clear, well-formatted natural-language answer including inline citations (string)
            "citations" : a de-duplicated array of source strings, e.g.
                            ["Document Name, Page 3", "Other Doc, Page 7"]
                            Use [] if no reliable sources are available.

            Strict JSON rules: no trailing commas, all strings properly escaped, no comments.

            Question: {query}

            DOCUMENTS:
            {context}

            JSON response:"""

            raw_response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            raw_text = raw_response.choices[0].message.content
            self.logger.debug(
                f"Raw LLM output (generate_cited_answer): {raw_text[:300]}"
            )

            result = self._parse_llm_json(raw_text)

            if not result["citations"]:
                self.logger.info(
                    "generate_cited_answer: citations array empty; "
                    "synthesising from node metadata"
                )
                seen_refs: set = set()
                synthesised: List[str] = []
                for node in all_nodes_seen:
                    doc  = node.get("_doc_name", "Unknown Document")
                    page = node.get("page_index", "?")
                    ref  = f"{doc}, Page {page}"
                    if ref not in seen_refs:
                        seen_refs.add(ref)
                        synthesised.append(ref)
                result["citations"] = synthesised

            self.logger.info(
                f"generate_cited_answer: done — "
                f"{len(result['citations'])} citation(s)"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"generate_cited_answer: unexpected error: {str(e)}", exc_info=True
            )
            return fallback

# QUERY EXECUTION PIPELINE

 # Unified query execution pipeline
class QueryExecutor:
        
    # Encapsulates the 3-step query process:
    # 1. Document verification
    # 2. Length analysis and content extraction
    # 3. Answer generation
    
    def __init__(self, api: PageIndexAPI, rag: RAGEngine):
        self.api = api
        self.rag = rag
        self.logger = api.logger
    
    def execute_query(
        self,
        query: str,
        selected_docs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        steps_log = []
        
        def log(step: str, msg: str):
            steps_log.append(f"[{step}] {msg}")
            if progress_callback:
                progress_callback(step, msg)
            self.logger.info(f"[{step}] {msg}")
        
        try:
        
            log("STEP_1", "Verifying document processing status...")
            
            verified_docs = []
            for doc in selected_docs:
                doc_id = doc.get("id")
                doc_name = doc.get("name", "Unknown")
                
                metadata = self.api.get_document_metadata(doc_id)
                if not metadata:
                    log("STEP_1", f"{doc_name}: Failed to retrieve metadata")
                    continue
                
                status = metadata.get("status", "unknown")
                page_count = metadata.get("page_count", 0)
                
                if status != "completed":
                    log("STEP_1", f"{doc_name}: Still processing (Status: {status})")
                    continue
                
                log("STEP_1", f"{doc_name}: Ready ({page_count} pages)")
                
                verified_docs.append({
                    "id": doc_id,
                    "name": doc_name,
                    "page_count": page_count,
                    "metadata": metadata
                })
            
            if not verified_docs:
                log("STEP_1", "No documents passed verification")
                return {
                    "success": False,
                    "answer": None,
                    "context_by_doc": {},
                    "steps_log": steps_log,
                    "error": "No documents are ready for querying. Please wait for processing to complete."
                }
            
        
            log("STEP_2", "Analyzing document length and determining strategy...")
            
            context_by_doc = {}
            
            for doc in verified_docs:
                doc_id = doc.get("id")
                doc_name = doc.get("name")
                page_count = doc.get("page_count", 0)
                
                log("STEP_2", f"Processing '{doc_name}' ({page_count} pages)...")
                
                # Fetch tree
                tree = self.api.get_tree(doc_id)
                if not tree:
                    log("STEP_2", f"Failed to fetch structure for {doc_name}")
                    continue
                
                if page_count <= 20:
                    # STRATEGY 1: SHORT DOCUMENTS (≤20 pages)
                    log("STEP_2", f"Strategy: FULL EXTRACTION ({page_count} pages ≤ 20)")
                    
                    all_nodes = self.rag.extract_full_content(tree)
                    log("STEP_2", f"Extracted {len(all_nodes)} content nodes")
                    
                    context_by_doc[doc_name] = {
                        "doc_id": doc_id,
                        "nodes": all_nodes,
                        "page_count": page_count,
                        "strategy": f"Full document extraction ({page_count} pages)"
                    }
                    
                    assets = self.rag.extract_asset_references(all_nodes)
                    if any(assets.values()):
                        for atype, refs in assets.items():
                            if refs:
                                log("STEP_2", f"Assets – {atype}: {len(refs)} reference(s)")
                
                else:
                    # STRATEGY 2: LONG DOCUMENTS (>20 pages)
                    log("STEP_2", f"Strategy: SELECTIVE EXTRACTION ({page_count} pages > 20)")
                    
                    log("STEP_2", "Analyzing document structure...")
                    structure = self.rag.extract_document_structure(tree)
                    
                    log("STEP_2", "Identifying relevant sections for your query...")
                    section_result = self.rag.identify_relevant_sections(query, structure, doc_name)
                    
                    if not section_result:
                        log("STEP_2", f"Failed to identify relevant sections in {doc_name}")
                        continue
                    
                    relevant_sections = section_result.get("sections", [])
                    log("STEP_2", f"Identified {len(relevant_sections)} relevant sections")
                    
                    # Use tree search to find relevant content
                    log("STEP_2", "Tree searching for relevant content...")
                    search_result = self.rag.llm_tree_search(query, tree)
                    
                    if not search_result:
                        log("STEP_2", f"Failed to search tree for {doc_name}")
                        continue
                    
                    node_ids = search_result.get("node_list", [])
                    if not node_ids:
                        log("STEP_2", f"No relevant nodes found")
                        continue
                    
                    # Retrieve the selected nodes
                    nodes = self.rag.find_nodes_by_ids(tree, node_ids)
                    log("STEP_2", f"Retrieved {len(nodes)} content nodes")
                    
                    context_by_doc[doc_name] = {
                        "doc_id": doc_id,
                        "nodes": nodes,
                        "page_count": page_count,
                        "strategy": f"Selective extraction ({len(relevant_sections)} sections from {page_count} pages)"
                    }
                    
                    assets = self.rag.extract_asset_references(nodes)
                    if any(assets.values()):
                        for atype, refs in assets.items():
                            if refs:
                                log("STEP_2", f"Assets – {atype}: {len(refs)} reference(s)")
            
            if not context_by_doc:
                log("STEP_2", "No content retrieved from any document")
                return {
                    "success": False,
                    "answer": None,
                    "context_by_doc": {},
                    "steps_log": steps_log,
                    "error": "No content could be retrieved from the selected documents."
                }
            
        
            log("STEP_3", "Generating cited answer...")

            answer_json = self.rag.generate_cited_answer(query, context_by_doc)

            response_text = answer_json.get("response", "")
            citations     = answer_json.get("citations", [])

            is_fallback = (
                response_text == RAGEngine._FALLBACK_JSON["response"]
                and not citations
            )

            if not is_fallback and response_text.strip():
                log("STEP_3", "Answer generated successfully")
                log("STEP_3", f"{len(citations)} citation(s) included")
                return {
                    "success":       True,
                    "answer":        response_text,
                    "citations":     citations,
                    "answer_json":   answer_json,
                    "context_by_doc": context_by_doc,
                    "steps_log":     steps_log,
                    "error":         None
                }
            else:
                log("STEP_3", "Failed to generate a useful answer")
                return {
                    "success":       False,
                    "answer":        None,
                    "citations":     [],
                    "answer_json":   answer_json,
                    "context_by_doc": context_by_doc,
                    "steps_log":     steps_log,
                    "error":         "Failed to generate answer. Please check the logs."
                }
        
        except Exception as e:
            log("ERROR", f"Unexpected error: {str(e)}")
            self.logger.error(f"Query execution error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "answer": None,
                "context_by_doc": {},
                "steps_log": steps_log,
                "error": f"An unexpected error occurred: {str(e)}"
            }