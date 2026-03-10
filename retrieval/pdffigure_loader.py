import json
import os


def normalize_path(path):
    """
    Normalize file paths so that Windows paths work inside WSL/Linux.

    pdffigures2 often outputs Windows-style paths such as:
        C:\\Users\\user\\figures\\figure1.png

    When running in WSL or Linux environments, the path must be converted
    into a mount path:

        /mnt/c/Users/user/figures/figure1.png

    Parameters
    ----------
    path : str
        Path string extracted from pdffigures2 JSON.

    Returns
    -------
    str
        Normalized path usable inside the current runtime environment.
    """

    # Already normalized WSL path
    if path.startswith("/mnt/c/"):
        return path

    # Convert Windows path to WSL path
    if path.startswith("C:\\"):
        return path.replace("C:\\", "/mnt/c/").replace("\\", "/")

    # Otherwise return unchanged
    return path


def load_pdffigures(json_path):
    """
    Load figures extracted from pdffigures2.

    pdffigures2 outputs a JSON file containing metadata about figures
    detected in a PDF. This function parses that JSON and prepares
    a structured representation of each figure for downstream tasks
    such as visual retrieval or diagram generation.

    Each figure contains:
        - figure type (figure/table/etc.)
        - caption text
        - OCR or embedded text within the image
        - normalized image path

    Parameters
    ----------
    json_path : str
        Path to the pdffigures2 JSON output.

    Returns
    -------
    list[dict]
        List of figure dictionaries containing:

        {
            "type": str,
            "caption": str,
            "text": str,
            "image": str,
            "raw_text": list[str]
        }
    """

    with open(json_path) as f:
        data = json.load(f)

    figures = []

    for item in data:

        # Extract metadata fields
        caption = item.get("caption", "")
        fig_type = item.get("figType", "")
        img_path = normalize_path(item.get("renderURL", ""))

        # OCR / detected text inside figure
        image_text = " ".join(item.get("imageText", []))

        # Skip figures if the image file does not exist
        if not os.path.exists(img_path):
            print("IMAGE NOT FOUND:", img_path)
            continue

        # Store structured figure representation
        figures.append({
            "type": fig_type,
            "caption": caption,

            # Combined textual description used for retrieval
            "text": caption + " " + image_text,

            # Image file path
            "image": img_path,

            # Raw OCR tokens
            "raw_text": item.get("imageText", [])
        })

    return figures