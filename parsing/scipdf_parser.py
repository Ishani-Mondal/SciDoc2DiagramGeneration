

import scipdf
from utils.logger import log
from utils.io_utils import save_json


class SciPDFParser:
    """
    Parser for extracting structured content from scientific PDFs.

    This class uses the `scipdf` library (which internally relies on
    the GROBID service) to convert a PDF document into a structured
    Python dictionary containing sections, figures, and tables.

    The parsed content is then cleaned and returned in a simplified
    format suitable for downstream components such as:

        • retrieval
        • diagram planning
        • evidence extraction
        • question answering over document content

    Attributes
    ----------
    grobid_url : str
        URL of the running GROBID service used by the SciPDF parser
        to extract structured text from PDFs.
    """

    def __init__(self):
        """
        Initialize the SciPDF parser.

        The parser requires a running GROBID server to process PDFs.
        By default, it assumes the GROBID service is running locally
        on port 8070.
        """

        # Local GROBID service used for parsing PDF structure
        self.grobid_url = "http://127.0.0.1:8070"

    def parse(self, pdf_path):
        """
        Parse a scientific PDF into structured document components.

        This method processes a PDF file and extracts key elements
        including:

            • section text
            • figure captions
            • table captions

        The extracted elements are stored as lists and returned as a
        structured dictionary for downstream tasks.

        Parameters
        ----------
        pdf_path : str
            Path to the input PDF file to be parsed.

        Returns
        -------
        dict
            Dictionary containing parsed document components:

            {
                "sections": list[str],   # textual body sections
                "figures": list[str],    # figure captions
                "tables": list[str]      # table captions
            }

        Notes
        -----
        - If parsing fails, the function returns an empty document
          structure to prevent downstream pipeline failures.
        - Extracted content is also saved to disk for debugging and
          inspection.
        """

        # --------------------------------------------------
        # Begin PDF parsing
        # --------------------------------------------------
        log("Parsing PDF with SciPDF")

        try:
            # Use SciPDF (backed by GROBID) to parse the PDF
            parsed = scipdf.parse_pdf_to_dict(
                pdf_path,
                grobid_url=self.grobid_url
            )

        except Exception as e:
            # If parsing fails, log the error
            log(f"SciPDF parsing failed: {e}")
            parsed = None

        # --------------------------------------------------
        # Handle parsing failure
        # --------------------------------------------------
        if parsed is None:
            log("WARNING: SciPDF returned None. Using empty document.")

            return {
                "sections": [],
                "figures": [],
                "tables": []
            }

        # --------------------------------------------------
        # Containers for parsed document elements
        # --------------------------------------------------
        sections = []
        figures = []
        tables = []

        # --------------------------------------------------
        # Extract section text
        # --------------------------------------------------
        # Each section contains structured metadata including text.
        for sec in parsed.get("sections", []):
            text = sec.get("text")

            if text:
                sections.append(text)

        # --------------------------------------------------
        # Extract figure captions
        # --------------------------------------------------
        # Figures may contain either "caption" or "content".
        for fig in parsed.get("figures", []):
            caption = fig.get("caption") or fig.get("content")

            if caption:
                figures.append(caption)

        # --------------------------------------------------
        # Extract table captions
        # --------------------------------------------------
        for tab in parsed.get("tables", []):
            caption = tab.get("caption") or tab.get("content")

            if caption:
                tables.append(caption)

        # --------------------------------------------------
        # Save parsed results for debugging and logging
        # --------------------------------------------------
        save_json("parsed_sections", sections)
        save_json("parsed_figures", figures)
        save_json("parsed_tables", tables)

        log(f"Parsed {len(sections)} sections")

        # --------------------------------------------------
        # Return structured document representation
        # --------------------------------------------------
        return {
            "sections": sections,
            "figures": figures,
            "tables": tables
        }