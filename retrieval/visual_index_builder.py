def build_visual_docs(figures, extractor):
    """
    Build visual retrieval documents from extracted PDF figures.

    This function processes each figure using a vision-based extractor
    (e.g., Gemini Vision or another multimodal model) to extract
    structured data such as:

        - data series from charts
        - table rows
        - visual summaries

    The extracted information is converted into textual summaries
    which can be indexed by the semantic retriever.

    Parameters
    ----------
    figures : list[dict]
        List of figure objects produced by `load_pdffigures`.

        Each figure contains:
            {
                "caption": str,
                "image": str,
                "text": str,
                ...
            }

    extractor : callable
        Function that performs visual extraction.

        Signature:
            extractor(image_path, caption) -> dict

        Expected output format:
            {
                "series": {label: values},
                "rows": [...],
                ...
            }

    Returns
    -------
    list[dict]
        Visual documents used for semantic indexing:

        {
            "text": str,   # textual representation of visual data
            "image": str,  # path to figure image
            "data": dict   # structured extracted data
        }
    """

    visual_docs = []

    for f in figures:

        image = f["image"]
        caption = f["caption"]

        print("Processing:", image)

        # Extract structured data using a multimodal model
        data = extractor(image, caption)

        print("Gemini output:", data)

        # Skip if extraction fails
        if data is None:
            continue

        # Build textual summary used for embedding retrieval
        summary = caption + "\n"

        # Add chart series if available
        if data.get("series"):
            for k, v in data["series"].items():
                summary += f"{k}: {v}\n"

        # Add table rows if available
        if data.get("rows"):
            summary += str(data["rows"])

        # Store visual document
        visual_docs.append({
            "text": summary,
            "image": image,
            "data": data
        })

    return visual_docs