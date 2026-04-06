# %% [markdown]
# # Semiconductor Annual Report Research Pipeline
#
# This notebook builds a reusable research workflow for annual report PDFs in a semiconductor
# supply-chain thesis project. The pipeline is designed to:
#
# - scan a folder of company subfolders and PDF files
# - parse company names and fiscal years from flexible file naming patterns
# - extract PDF text page by page
# - chunk the text into traceable passages
# - build embeddings and a persistent FAISS index
# - run a predefined retrieval query set across the corpus
# - deduplicate retrieved passages
# - classify retrieved passages with an LLM using strict JSON validation
# - keep the original passage text unchanged
# - export final relevant passages and labels to Excel
#
# The notebook prioritizes clarity, robustness, and easy modification over clever abstractions.
# Every major step is commented so you can understand and adapt it later.

# %% [markdown]
# ## 1. Project Overview
#
# The workflow in this notebook follows a retrieval-then-classification design:
#
# 1. Discover all PDF files recursively under a base reports folder.
# 2. Parse metadata such as `firm_name` and `fiscal_year`.
# 3. Extract page-level text from each PDF.
# 4. Split pages into manageable chunks while preserving traceability.
# 5. Encode chunks with a sentence-transformer embedding model.
# 6. Save and reuse a local FAISS vector index for fast reruns.
# 7. Run a configurable query set across the whole corpus.
# 8. Deduplicate overlapping retrieval results.
# 9. Send each unique retrieved chunk to an LLM only for classification.
# 10. Drop chunks marked irrelevant.
# 11. Export the final structured evidence table to Excel.
#
# Important design choice:
# The original chunk text is never rewritten by the LLM. The `passage` column always contains
# the original extracted chunk text.

# %% [markdown]
# ## 2. Imports and Setup
#
# This cell imports the libraries used throughout the notebook. If a package is missing in your
# local environment, install it before running the notebook. Typical installs might include:
#
# ```bash
# pip install pandas openpyxl pymupdf sentence-transformers faiss-cpu tqdm openai
# ```
#
# If you prefer a different PDF parser, vector store, or LLM provider, you can swap those parts
# later in the helper functions and configuration section.

# %%
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import textwrap
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from IPython.display import display
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# Configure logging once near the top of the notebook.
# Logging helps us continue processing while still recording warnings and failures.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("annual_report_pipeline")


# Make pandas output easier to inspect in a notebook.
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 160)

# %% [markdown]
# ## 3. Configuration
#
# This is the main configuration block you will edit most often.
# Keep all high-level runtime choices here so the rest of the notebook stays stable.
#
# You can use this section to:
#
# - change the input and output folders
# - tune chunk size and retrieval depth
# - switch embedding or LLM models
# - rebuild or reuse an existing vector index
# - limit the run for debugging

# %%
CONFIG: Dict[str, Any] = {
    # Base project paths
    "base_input_folder": Path("data/reports"),
    "output_folder": Path("outputs"),
    "intermediate_folder": Path("outputs/intermediate"),
    "vector_store_path": Path("outputs/vector_store"),
    "excel_output_name": "semiconductor_thesis_passages.xlsx",
    # Debugging and run control
    "max_files": None,  # Example: 5 for quick testing
    "debug_company": None,  # Example: "Intel"
    "debug_year": None,  # Example: 2023
    "rebuild_index": False,
    "skip_existing_pdf_text": False,
    # Chunking controls
    "chunk_size_chars": 2200,
    "chunk_overlap_chars": 300,
    "min_chunk_chars": 250,
    # Retrieval controls
    "top_k_retrieval": 8,
    # Models
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "openai_model_name": "gpt-4.1-mini",
    # API key loading
    "openai_api_key_env_var": "OPENAI_API_KEY",
    # Persistence filenames
    "manifest_filename": "file_manifest.csv",
    "pages_filename": "pdf_pages.pkl",
    "chunks_filename": "chunks.pkl",
    "retrieval_filename": "retrieved_chunks.pkl",
    "classification_filename": "classified_chunks.pkl",
    "final_export_filename": "final_relevant_chunks.pkl",
}


# Retrieval queries are defined in a single dictionary for easy maintenance.
QUERY_SET: Dict[str, str] = {
    "E1 Operational Buffers": (
        "semiconductor company increasing safety stock inventory buffer capacity supply shortage risk"
    ),
    "E2 Footprint Diversification": (
        "semiconductor manufacturing geographic diversification new fab location nearshoring second source region"
    ),
    "E3 Supply Option Diversification": (
        "semiconductor dual sourcing alternative supplier critical materials wafer substrate backup supply"
    ),
    "E4 Robust Distribution": (
        "semiconductor logistics resilience distribution network freight transport disruption flexibility"
    ),
    "E5 Product Standardisation": (
        "semiconductor platform architecture product standardisation SKU rationalisation common design supply complexity"
    ),
    "E6 Partner Network Strengthening": (
        "semiconductor long-term supply agreement customer lock-in capacity reservation partnership foundry collaboration"
    ),
    "E7 SCRM and Visibility": (
        "semiconductor supply chain risk management visibility monitoring business continuity geopolitical risk assessment"
    ),
    "Broad geopolitical query": (
        "semiconductor geopolitical risk export controls tariffs supply chain disruption government policy CHIPS Act"
    ),
}


# Create output folders up front so later cells can save intermediate results without failing.
CONFIG["output_folder"].mkdir(parents=True, exist_ok=True)
CONFIG["intermediate_folder"].mkdir(parents=True, exist_ok=True)
CONFIG["vector_store_path"].mkdir(parents=True, exist_ok=True)


# Print the active configuration so you can verify it before running the full workflow.
print("Active configuration:")
for key, value in CONFIG.items():
    print(f"- {key}: {value}")

# %% [markdown]
# ## 4. Helper Functions
#
# This section contains the reusable helper functions that power the pipeline.
# The functions are kept modular but intentionally straightforward so that each step
# remains easy to inspect and replace later.

# %%
@dataclass
class FileRecord:
    """Metadata for one discovered PDF file."""

    firm_name: Optional[str]
    fiscal_year: Optional[int]
    source_file: str
    folder_name: str
    file_name: str


@dataclass
class PageRecord:
    """Extracted text and metadata for one PDF page."""

    firm_name: Optional[str]
    fiscal_year: Optional[int]
    source_file: str
    page_number: int
    section: Optional[str]
    page_text: str


@dataclass
class ChunkRecord:
    """Chunk-level record preserved through retrieval and classification."""

    firm_name: Optional[str]
    fiscal_year: Optional[int]
    source_file: str
    page_number: int
    section: Optional[str]
    chunk_id: str
    passage: str


EXPECTED_CLASSIFICATION_KEYS: List[str] = [
    "relevant",
    "strategy_category",
    "secondary_category",
    "geopolitical_trigger",
    "orientation",
    "main_point",
    "confidence",
]


def save_pickle(obj: Any, path: Path) -> None:
    """Save a Python object to disk using pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object from disk."""

    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


def normalise_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving readable paragraph breaks."""

    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_year_from_text(text: str) -> Optional[int]:
    """Extract the first plausible 4-digit fiscal year from a string."""

    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    return int(match.group(0))


def clean_company_name(raw_name: str) -> str:
    """Clean company name strings parsed from folders or filenames."""

    cleaned = re.sub(r"[_\-]+", " ", raw_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_firm_and_year_from_path(pdf_path: Path) -> FileRecord:
    """Parse company name and fiscal year, preferring the filename over the folder name.

    Strategy:
    1. Try to parse the year from the filename stem first.
    2. Remove that year and common separators from the stem to infer company name.
    3. If filename parsing fails, fall back to the immediate parent folder name.
    4. Log a warning if a fiscal year cannot be identified.
    """

    folder_name = pdf_path.parent.name
    file_name = pdf_path.name
    stem = pdf_path.stem

    fiscal_year = extract_year_from_text(stem)

    firm_name_from_filename = re.sub(r"(19|20)\d{2}", " ", stem)
    firm_name_from_filename = re.sub(r"annual\s*report", " ", firm_name_from_filename, flags=re.IGNORECASE)
    firm_name_from_filename = clean_company_name(firm_name_from_filename)
    firm_name_from_filename = firm_name_from_filename or None

    firm_name_from_folder = clean_company_name(folder_name) if folder_name else None
    firm_name = firm_name_from_filename or firm_name_from_folder

    if fiscal_year is None:
        fiscal_year = extract_year_from_text(folder_name)

    if fiscal_year is None:
        logger.warning("Could not parse fiscal year for file: %s", pdf_path)

    return FileRecord(
        firm_name=firm_name,
        fiscal_year=fiscal_year,
        source_file=str(pdf_path.resolve()),
        folder_name=folder_name,
        file_name=file_name,
    )


def discover_pdf_files(
    base_folder: Path,
    max_files: Optional[int] = None,
    debug_company: Optional[str] = None,
    debug_year: Optional[int] = None,
) -> pd.DataFrame:
    """Recursively discover PDF files and parse metadata into a manifest DataFrame."""

    pdf_paths = sorted(base_folder.rglob("*.pdf"))
    records: List[FileRecord] = []

    for pdf_path in pdf_paths:
        record = parse_firm_and_year_from_path(pdf_path)

        if debug_company and (record.firm_name or "").lower() != debug_company.lower():
            continue

        if debug_year and record.fiscal_year != debug_year:
            continue

        records.append(record)

        if max_files is not None and len(records) >= max_files:
            break

    manifest_df = pd.DataFrame([asdict(record) for record in records])

    if manifest_df.empty:
        logger.warning("No PDF files found under %s", base_folder.resolve())
    else:
        logger.info("Discovered %s PDF files", len(manifest_df))

    return manifest_df


def detect_section_heading(page_text: str) -> Optional[str]:
    """Attempt a lightweight section heading guess from the top of a page.

    This heuristic is intentionally conservative. If a reliable heading is not found,
    the function returns `None` rather than guessing too aggressively.
    """

    candidate_lines: List[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if stripped:
            candidate_lines.append(stripped)
        if len(candidate_lines) >= 8:
            break

    for line in candidate_lines:
        is_reasonable_length = 3 <= len(line) <= 120
        looks_like_heading = (
            line.isupper()
            or bool(re.match(r"^[A-Z][A-Za-z0-9/&,\-\s]{2,}$", line))
            or bool(re.match(r"^\d+(\.\d+)*\s+[A-Z].+", line))
        )
        low_punctuation_density = line.count(".") <= 1 and line.count(":") <= 1

        if is_reasonable_length and looks_like_heading and low_punctuation_density:
            return line

    return None


def extract_pdf_pages(pdf_path: Path, fallback_record: FileRecord) -> List[PageRecord]:
    """Extract page-level text from a PDF using PyMuPDF.

    Unreadable PDFs are skipped with a warning rather than stopping the notebook.
    """

    page_records: List[PageRecord] = []

    try:
        with fitz.open(pdf_path) as document:
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                page_text = normalise_whitespace(page.get_text("text"))

                # Empty pages are skipped to keep the downstream pipeline cleaner.
                if not page_text:
                    continue

                section = detect_section_heading(page_text)
                page_records.append(
                    PageRecord(
                        firm_name=fallback_record.firm_name,
                        fiscal_year=fallback_record.fiscal_year,
                        source_file=fallback_record.source_file,
                        page_number=page_index + 1,
                        section=section,
                        page_text=page_text,
                    )
                )
    except Exception as error:  # noqa: BLE001 - we want resilience here
        logger.warning("Failed to read PDF %s: %s", pdf_path, error)

    return page_records


def split_text_into_chunks(
    text: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> List[str]:
    """Split a page of text into overlapping chunks using paragraph-friendly boundaries.

    The logic favors readability and traceability:
    - paragraphs are kept together when possible
    - chunks overlap slightly to reduce boundary loss
    - very small chunks are dropped
    """

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        proposed = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

        if len(proposed) <= chunk_size_chars:
            current_chunk = proposed
            continue

        if current_chunk and len(current_chunk) >= min_chunk_chars:
            chunks.append(current_chunk.strip())

        # If a single paragraph is too large, split it more aggressively.
        if len(paragraph) > chunk_size_chars:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size_chars
                piece = paragraph[start:end].strip()
                if len(piece) >= min_chunk_chars:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(end - chunk_overlap_chars, start + 1)
            current_chunk = ""
        else:
            current_chunk = paragraph

    if current_chunk and len(current_chunk) >= min_chunk_chars:
        chunks.append(current_chunk.strip())

    # A second pass adds overlap between adjacent chunks from the same page.
    overlapped_chunks: List[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped_chunks.append(chunk)
            continue

        previous_chunk = chunks[index - 1]
        overlap_prefix = previous_chunk[-chunk_overlap_chars:].strip()

        if overlap_prefix and not chunk.startswith(overlap_prefix):
            combined_chunk = f"{overlap_prefix}\n\n{chunk}".strip()
        else:
            combined_chunk = chunk

        overlapped_chunks.append(combined_chunk[: chunk_size_chars + chunk_overlap_chars].strip())

    return overlapped_chunks


def build_chunk_id(
    firm_name: Optional[str],
    fiscal_year: Optional[int],
    source_file: str,
    page_number: int,
    passage: str,
) -> str:
    """Create a deterministic chunk ID from stable metadata plus the passage text."""

    raw_value = "||".join(
        [
            str(firm_name or ""),
            str(fiscal_year or ""),
            source_file,
            str(page_number),
            passage,
        ]
    )
    return sha1(raw_value.encode("utf-8")).hexdigest()


def create_chunks_from_pages(
    pages_df: pd.DataFrame,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> pd.DataFrame:
    """Create chunk-level records from page-level text while preserving metadata."""

    chunk_records: List[ChunkRecord] = []

    for page_row in tqdm(
        pages_df.itertuples(index=False),
        total=len(pages_df),
        desc="Chunking pages",
    ):
        page_chunks = split_text_into_chunks(
            text=page_row.page_text,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )

        for chunk_text in page_chunks:
            chunk_id = build_chunk_id(
                firm_name=page_row.firm_name,
                fiscal_year=page_row.fiscal_year,
                source_file=page_row.source_file,
                page_number=page_row.page_number,
                passage=chunk_text,
            )
            chunk_records.append(
                ChunkRecord(
                    firm_name=page_row.firm_name,
                    fiscal_year=page_row.fiscal_year,
                    source_file=page_row.source_file,
                    page_number=page_row.page_number,
                    section=page_row.section,
                    chunk_id=chunk_id,
                    passage=chunk_text,
                )
            )

    chunks_df = pd.DataFrame([asdict(record) for record in chunk_records])

    if not chunks_df.empty:
        chunks_df = chunks_df.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)

    return chunks_df


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model once so it can be reused for indexing and retrieval."""

    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode text passages as float32 vectors suitable for FAISS."""

    embeddings = embedding_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """Persist a FAISS index to disk."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""

    return faiss.read_index(str(index_path))


def build_or_load_vector_store(
    chunks_df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    vector_store_path: Path,
    rebuild_index: bool,
) -> Tuple[faiss.Index, pd.DataFrame]:
    """Build or reuse a persistent FAISS vector index plus aligned metadata."""

    index_file = vector_store_path / "chunks.faiss"
    metadata_file = vector_store_path / "chunk_metadata.pkl"

    if not rebuild_index and index_file.exists() and metadata_file.exists():
        logger.info("Loading existing vector store from disk")
        index = load_faiss_index(index_file)
        stored_metadata = load_pickle(metadata_file)
        return index, stored_metadata

    if chunks_df.empty:
        raise ValueError("Cannot build vector store because chunks_df is empty.")

    logger.info("Building a new vector store with %s chunks", len(chunks_df))
    embeddings = embed_texts(chunks_df["passage"].tolist(), embedding_model)

    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    save_faiss_index(index, index_file)
    save_pickle(chunks_df.reset_index(drop=True), metadata_file)

    return index, chunks_df.reset_index(drop=True)


def retrieve_top_k_chunks(
    query_set: Dict[str, str],
    embedding_model: SentenceTransformer,
    vector_index: faiss.Index,
    chunk_metadata_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Run all configured queries against the vector index and collect top-k results."""

    retrieval_rows: List[Dict[str, Any]] = []

    for query_label, query_text in tqdm(query_set.items(), desc="Running retrieval queries"):
        query_vector = embed_texts([query_text], embedding_model)
        scores, indices = vector_index.search(query_vector, top_k)

        for rank, (score, chunk_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_idx < 0:
                continue

            chunk_row = chunk_metadata_df.iloc[int(chunk_idx)].to_dict()
            retrieval_rows.append(
                {
                    **chunk_row,
                    "retrieval_query": query_label,
                    "retrieval_query_text": query_text,
                    "retrieval_rank": rank,
                    "retrieval_score": float(score),
                }
            )

    retrieval_df = pd.DataFrame(retrieval_rows)
    return retrieval_df


def deduplicate_retrieved_chunks(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate retrieved results so each chunk is classified only once.

    We keep the best retrieval instance of each chunk based on score and rank.
    """

    if retrieval_df.empty:
        return retrieval_df.copy()

    deduped_df = (
        retrieval_df.sort_values(
            by=["chunk_id", "retrieval_score", "retrieval_rank"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["chunk_id"], keep="first")
        .reset_index(drop=True)
    )
    return deduped_df


def get_openai_client(api_key_env_var: str) -> OpenAI:
    """Create an OpenAI client using an API key stored in an environment variable."""

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env_var}' is not set. "
            "Set your API key before running the classification step."
        )
    return OpenAI(api_key=api_key)


def build_classification_prompt(passage_row: pd.Series) -> str:
    """Create the classification prompt.

    The prompt explicitly instructs the model to classify only and never rewrite the passage.
    """

    passage = passage_row["passage"]
    metadata_text = textwrap.dedent(
        f"""
        Metadata:
        - firm_name: {passage_row.get("firm_name")}
        - fiscal_year: {passage_row.get("fiscal_year")}
        - source_file: {passage_row.get("source_file")}
        - page_number: {passage_row.get("page_number")}
        - section: {passage_row.get("section")}
        - retrieval_query: {passage_row.get("retrieval_query")}
        """
    ).strip()

    instructions = textwrap.dedent(
        """
        You are classifying an original passage from an annual report for a semiconductor supply-chain thesis project.

        Important rules:
        1. Do not rewrite, summarize, or modify the passage.
        2. Your task is classification only.
        3. Return exactly one JSON object.
        4. The JSON object must contain exactly these keys:
           relevant
           strategy_category
           secondary_category
           geopolitical_trigger
           orientation
           main_point
           confidence
        5. Use null when a field cannot be inferred confidently, except for:
           - relevant: must be true or false
           - confidence: must be a number between 0 and 1
        6. If the passage is not materially relevant to supply-chain resilience, sourcing,
           manufacturing footprint, logistics robustness, product standardisation,
           partnerships, risk monitoring, or geopolitical exposure, return relevant=false.
        7. Do not include markdown, explanation text, or code fences.
        """
    ).strip()

    return f"{instructions}\n\n{metadata_text}\n\nPassage:\n{passage}"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response string."""

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_classification_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean the structured classification output."""

    if set(payload.keys()) != set(EXPECTED_CLASSIFICATION_KEYS):
        raise ValueError(
            f"Invalid JSON keys. Expected exactly {EXPECTED_CLASSIFICATION_KEYS}, got {list(payload.keys())}"
        )

    if not isinstance(payload["relevant"], bool):
        raise ValueError("'relevant' must be a boolean.")

    confidence_value = payload["confidence"]
    if not isinstance(confidence_value, (int, float)):
        raise ValueError("'confidence' must be numeric.")
    if not 0 <= float(confidence_value) <= 1:
        raise ValueError("'confidence' must be between 0 and 1.")

    cleaned_payload: Dict[str, Any] = {
        "relevant": payload["relevant"],
        "strategy_category": payload["strategy_category"],
        "secondary_category": payload["secondary_category"],
        "geopolitical_trigger": payload["geopolitical_trigger"],
        "orientation": payload["orientation"],
        "main_point": payload["main_point"],
        "confidence": float(confidence_value),
    }

    return cleaned_payload


def request_openai_json_response(
    client: OpenAI,
    model_name: str,
    prompt: str,
) -> str:
    """Request a raw text response from OpenAI.

    This function is intentionally isolated so the LLM provider can be swapped later.
    """

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return response.output_text


def classify_chunk_with_retries(
    passage_row: pd.Series,
    client: OpenAI,
    model_name: str,
    max_attempts: int = 2,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Classify one chunk with validation and one repair retry if needed."""

    prompt = build_classification_prompt(passage_row)
    raw_response_text = ""

    for attempt_number in range(1, max_attempts + 1):
        try:
            if attempt_number == 1:
                raw_response_text = request_openai_json_response(client, model_name, prompt)
            else:
                repair_prompt = textwrap.dedent(
                    f"""
                    The following output was invalid. Repair it into valid JSON only.

                    Required keys:
                    {EXPECTED_CLASSIFICATION_KEYS}

                    Rules:
                    - Return exactly one JSON object.
                    - No markdown or commentary.
                    - Keep the intended classification meaning if possible.
                    - relevant must be boolean.
                    - confidence must be numeric between 0 and 1.

                    Invalid output:
                    {raw_response_text}
                    """
                ).strip()
                raw_response_text = request_openai_json_response(client, model_name, repair_prompt)

            parsed_payload = extract_json_object(raw_response_text)
            if parsed_payload is None:
                raise ValueError("No valid JSON object found in LLM response.")

            validated_payload = validate_classification_payload(parsed_payload)
            return validated_payload, None

        except Exception as error:  # noqa: BLE001 - resilience is more important here
            if attempt_number == max_attempts:
                failure_message = f"Classification failed after {max_attempts} attempts: {error}"
                return None, failure_message

    return None, "Unexpected classification failure."


def classify_retrieved_chunks(
    retrieved_df: pd.DataFrame,
    client: OpenAI,
    model_name: str,
) -> pd.DataFrame:
    """Classify each deduplicated retrieved chunk and keep structured failure logs."""

    if retrieved_df.empty:
        expected_columns = list(retrieved_df.columns) + [
            "classification_failed",
            "classification_error",
            *EXPECTED_CLASSIFICATION_KEYS,
        ]
        return pd.DataFrame(columns=expected_columns)

    classification_rows: List[Dict[str, Any]] = []

    for passage_row in tqdm(
        retrieved_df.itertuples(index=False),
        total=len(retrieved_df),
        desc="Classifying retrieved chunks",
    ):
        row_series = pd.Series(passage_row._asdict())
        classification_payload, failure_message = classify_chunk_with_retries(
            passage_row=row_series,
            client=client,
            model_name=model_name,
        )

        base_record = row_series.to_dict()

        if classification_payload is None:
            logger.warning(
                "Classification failed for chunk %s: %s",
                base_record["chunk_id"],
                failure_message,
            )
            classification_rows.append(
                {
                    **base_record,
                    "classification_failed": True,
                    "classification_error": failure_message,
                    "relevant": None,
                    "strategy_category": None,
                    "secondary_category": None,
                    "geopolitical_trigger": None,
                    "orientation": None,
                    "main_point": None,
                    "confidence": None,
                }
            )
            continue

        classification_rows.append(
            {
                **base_record,
                "classification_failed": False,
                "classification_error": None,
                **classification_payload,
            }
        )

    classification_df = pd.DataFrame(classification_rows)
    return classification_df


def build_final_export(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant rows and arrange the final export columns."""

    required_columns = [
        "firm_name",
        "fiscal_year",
        "section",
        "page_number",
        "passage",
        "strategy_category",
        "secondary_category",
        "geopolitical_trigger",
        "orientation",
        "main_point",
        "confidence",
        "source_file",
        "chunk_id",
        "retrieval_query",
        "retrieval_rank",
    ]

    if classified_df.empty or "relevant" not in classified_df.columns:
        return pd.DataFrame(columns=required_columns)

    final_df = classified_df.loc[classified_df["relevant"] == True].copy()  # noqa: E712

    # Drop duplicate rows conservatively using chunk identity and retained retrieval label.
    final_df = final_df.drop_duplicates(
        subset=["chunk_id", "retrieval_query", "retrieval_rank"]
    ).reset_index(drop=True)

    return final_df[required_columns]


def safe_value_counts(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    """Return a value count series even when a column is missing or empty."""

    if column_name not in dataframe.columns or dataframe.empty:
        return pd.Series(dtype="int64")
    return dataframe[column_name].fillna("Missing").value_counts(dropna=False)

# %% [markdown]
# ## 5. File Discovery and Metadata Parsing
#
# This cell scans the reports folder recursively, parses metadata, and saves a manifest.
# It is a good early checkpoint because it lets you confirm:
#
# - how many files were found
# - whether company names were parsed correctly
# - whether fiscal years were detected consistently

# %%
manifest_df = discover_pdf_files(
    base_folder=CONFIG["base_input_folder"],
    max_files=CONFIG["max_files"],
    debug_company=CONFIG["debug_company"],
    debug_year=CONFIG["debug_year"],
)

manifest_output_path = CONFIG["intermediate_folder"] / CONFIG["manifest_filename"]
manifest_df.to_csv(manifest_output_path, index=False)

print(f"Manifest saved to: {manifest_output_path.resolve()}")
display(manifest_df.head(10))

# %% [markdown]
# ## 6. PDF Text Extraction
#
# This cell reads each discovered PDF and extracts text page by page.
# Every page record keeps core metadata so that later chunks remain traceable back
# to the original file and page.
#
# Robustness notes:
#
# - unreadable files are skipped with warnings
# - empty pages are ignored
# - section headings are guessed only when the heuristic is reasonably confident

# %%
pages_output_path = CONFIG["intermediate_folder"] / CONFIG["pages_filename"]

if pages_output_path.exists() and CONFIG["skip_existing_pdf_text"]:
    logger.info("Loading cached page extraction from %s", pages_output_path)
    pages_df = load_pickle(pages_output_path)
else:
    page_records: List[PageRecord] = []

    for file_row in tqdm(
        manifest_df.itertuples(index=False),
        total=len(manifest_df),
        desc="Extracting PDF pages",
    ):
        source_path = Path(file_row.source_file)
        fallback_record = FileRecord(
            firm_name=file_row.firm_name,
            fiscal_year=file_row.fiscal_year,
            source_file=file_row.source_file,
            folder_name=file_row.folder_name,
            file_name=file_row.file_name,
        )
        page_records.extend(extract_pdf_pages(source_path, fallback_record))

    pages_df = pd.DataFrame([asdict(record) for record in page_records])
    save_pickle(pages_df, pages_output_path)

print(f"Page extraction saved to: {pages_output_path.resolve()}")
display(pages_df.head(10))

# %% [markdown]
# ## 7. Chunking
#
# This cell converts page text into manageable passages.
#
# Design choices:
#
# - chunking happens within pages to preserve page-level traceability
# - chunk sizes use character limits as a practical approximation to token length
# - overlap is added to reduce information loss at chunk boundaries
# - each chunk gets a deterministic `chunk_id` so reruns stay stable

# %%
chunks_output_path = CONFIG["intermediate_folder"] / CONFIG["chunks_filename"]

chunks_df = create_chunks_from_pages(
    pages_df=pages_df,
    chunk_size_chars=CONFIG["chunk_size_chars"],
    chunk_overlap_chars=CONFIG["chunk_overlap_chars"],
    min_chunk_chars=CONFIG["min_chunk_chars"],
)
save_pickle(chunks_df, chunks_output_path)

print(f"Chunks saved to: {chunks_output_path.resolve()}")
display(chunks_df.head(10))

# %% [markdown]
# ## 8. Embedding and Index Creation
#
# This cell loads the embedding model and either:
#
# - builds a new FAISS index from the current chunks, or
# - loads a previously saved index from disk
#
# The vector store is kept local and persistent so reruns are much faster when the corpus
# does not change.

# %%
embedding_model = load_embedding_model(CONFIG["embedding_model_name"])

vector_index, indexed_chunks_df = build_or_load_vector_store(
    chunks_df=chunks_df,
    embedding_model=embedding_model,
    vector_store_path=CONFIG["vector_store_path"],
    rebuild_index=CONFIG["rebuild_index"],
)

print("Vector store is ready.")
print(f"Indexed chunk count: {len(indexed_chunks_df):,}")

# %% [markdown]
# ## 9. Retrieval
#
# This cell runs the predefined query set against the full chunk corpus.
# Each retrieval result keeps:
#
# - the original chunk metadata
# - the query label that retrieved it
# - the rank within that query
# - the similarity score
#
# This allows you to trace why each passage entered the classification stage.

# %%
retrieval_output_path = CONFIG["intermediate_folder"] / CONFIG["retrieval_filename"]

retrieval_df = retrieve_top_k_chunks(
    query_set=QUERY_SET,
    embedding_model=embedding_model,
    vector_index=vector_index,
    chunk_metadata_df=indexed_chunks_df,
    top_k=CONFIG["top_k_retrieval"],
)
save_pickle(retrieval_df, retrieval_output_path)

print(f"Retrieval results saved to: {retrieval_output_path.resolve()}")
display(retrieval_df.head(10))

# %% [markdown]
# ## 10. Deduplication
#
# The same chunk can be retrieved by multiple queries.
# To avoid redundant LLM calls, this cell keeps only one best retrieval instance per chunk.
#
# The retained row is chosen using:
#
# - highest retrieval score first
# - best rank as a tiebreaker

# %%
deduplicated_retrieval_df = deduplicate_retrieved_chunks(retrieval_df)

print(f"Retrieved rows before deduplication: {len(retrieval_df):,}")
print(f"Retrieved rows after deduplication:  {len(deduplicated_retrieval_df):,}")
display(deduplicated_retrieval_df.head(10))

# %% [markdown]
# ## 11. LLM Classification
#
# This cell sends each deduplicated chunk to an LLM for structured classification only.
#
# Important safeguards:
#
# - the LLM is not asked to rewrite the passage
# - the original passage text stays untouched in the dataset
# - outputs must match the exact JSON schema expected by the notebook
# - invalid JSON triggers one repair retry
# - failures are logged and do not stop the run
#
# Before running this cell, make sure your API key is available in the configured
# environment variable.

# %%
classification_output_path = CONFIG["intermediate_folder"] / CONFIG["classification_filename"]

openai_client = get_openai_client(CONFIG["openai_api_key_env_var"])

classified_df = classify_retrieved_chunks(
    retrieved_df=deduplicated_retrieval_df,
    client=openai_client,
    model_name=CONFIG["openai_model_name"],
)
save_pickle(classified_df, classification_output_path)

print(f"Classification results saved to: {classification_output_path.resolve()}")
display(classified_df.head(10))

# %% [markdown]
# ## 12. Final Filtering and Export
#
# This cell filters out chunks where `relevant` is `false`, keeps the original passage text,
# and exports the final table to Excel.
#
# The Excel file contains both the thesis-relevant classification columns and the traceability
# fields needed to audit the original source.

# %%
final_output_path = CONFIG["output_folder"] / CONFIG["excel_output_name"]
final_pickle_path = CONFIG["intermediate_folder"] / CONFIG["final_export_filename"]

final_relevant_df = build_final_export(classified_df)

save_pickle(final_relevant_df, final_pickle_path)
final_relevant_df.to_excel(final_output_path, index=False, engine="openpyxl")

print(f"Final relevant rows saved to pickle: {final_pickle_path.resolve()}")
print(f"Final Excel export saved to:         {final_output_path.resolve()}")
display(final_relevant_df.head(10))

# %% [markdown]
# ## 13. Quality Checks
#
# This section gives a quick sanity-check overview of the run.
# It helps answer practical questions such as:
#
# - how many files were actually processed
# - how many chunks were generated
# - how many retrieval results survived deduplication
# - how many final relevant rows remain
# - whether some companies or years dominate the output
# - which strategy categories appear most often

# %%
files_processed = len(manifest_df)
chunks_created = len(chunks_df)
retrieved_chunks = len(retrieval_df)
deduplicated_chunks = len(deduplicated_retrieval_df)
final_relevant_rows = len(final_relevant_df)

print("Quality check summary")
print("---------------------")
print(f"Files processed:            {files_processed:,}")
print(f"Chunks created:             {chunks_created:,}")
print(f"Retrieved chunks:           {retrieved_chunks:,}")
print(f"Deduplicated chunks:        {deduplicated_chunks:,}")
print(f"Final relevant rows:        {final_relevant_rows:,}")
print()

print("Counts by company")
display(safe_value_counts(final_relevant_df, "firm_name").rename("count").to_frame())
print()

print("Counts by year")
display(safe_value_counts(final_relevant_df, "fiscal_year").rename("count").to_frame())
print()

print("Counts by strategy category")
display(safe_value_counts(final_relevant_df, "strategy_category").rename("count").to_frame())
print()

print("Sample rows for manual inspection")
sample_columns = [
    "firm_name",
    "fiscal_year",
    "page_number",
    "section",
    "retrieval_query",
    "strategy_category",
    "main_point",
    "confidence",
    "passage",
]
display(final_relevant_df[sample_columns].head(8))

# %% [markdown]
# ## 14. Optional Analysis Preview
#
# This final section is intentionally lightweight. It gives you a quick look at the resulting
# evidence table without trying to perform the full thesis analysis inside the same notebook.
#
# You can extend this section later with:
#
# - time-series plots by firm and year
# - category co-occurrence tables
# - comparisons between geopolitical and operational resilience patterns
# - export subsets for manual coding review

# %%
if not final_relevant_df.empty:
    preview_table = (
        final_relevant_df.groupby(["firm_name", "fiscal_year", "strategy_category"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["firm_name", "fiscal_year", "count"], ascending=[True, True, False])
    )
    display(preview_table.head(20))
else:
    print("No relevant rows were returned, so there is nothing to preview yet.")

# %% [markdown]
# ## Notes on Extension
#
# The notebook is intentionally structured so you can replace major components later:
#
# - PDF parser:
#   Replace `extract_pdf_pages()` if you want to use `pdfplumber` or OCR.
# - Embedding model:
#   Change `embedding_model_name` and, if needed, the model-loading function.
# - Vector store:
#   Swap the FAISS helpers for Chroma or another local index.
# - LLM provider:
#   Replace `request_openai_json_response()` and keep the validation logic unchanged.
#
# Recommended next checks before using the notebook on a full corpus:
#
# 1. Run on one company and one year with `debug_company` and `debug_year`.
# 2. Inspect chunk quality manually.
# 3. Review a sample of final Excel rows to confirm the categories match your thesis logic.
# 4. Adjust query wording and `top_k_retrieval` based on recall quality.
