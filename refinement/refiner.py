from llm.gpt_client import call_llm
from generation.code_generator import extract_python_code
from utils.io_utils import save_code


def refine_diagram(code, feedback, step):
    """
    Refine diagram generation code using critic feedback.

    This function updates the Python code used to generate a diagram
    by incorporating feedback produced by the evaluation critics
    (Completeness, Faithfulness, Layout).

    The refinement process is performed by sending the existing code
    and aggregated feedback to a language model, which generates an
    improved version of the code. The returned response is parsed to
    extract executable Python code, which is then saved for logging
    and reproducibility.

    Parameters
    ----------
    code : str
        The current Python code used to generate the diagram.
        This code typically produces a visualization (e.g., via
        matplotlib) and saves it as an image.

    feedback : list[str] or str
        Aggregated feedback from critic modules. This feedback may
        include suggestions such as:
            - missing elements in the diagram
            - incorrect values or labels
            - layout or readability improvements

    step : int
        The current refinement iteration number. This is used for:
            - tracking refinement progress
            - naming saved code versions.

    Returns
    -------
    str
        Refined Python code that incorporates the provided feedback
        and can be executed to regenerate the improved diagram.

    Notes
    -----
    The refinement workflow consists of the following steps:

    1. Construct a prompt that includes:
        - the existing diagram-generating code
        - feedback from evaluation critics

    2. Send the prompt to a language model using `call_llm`.

    3. Extract executable Python code from the LLM response using
       `extract_python_code`.

    4. Save the refined code to disk for reproducibility and debugging.

    5. Return the refined code so that it can be rendered into a new
       diagram in the next refinement iteration.

    The saved files follow the naming pattern:

        refined_code_{step}.py
    """

    # --------------------------------------------------
    # Build refinement prompt for the LLM
    # --------------------------------------------------
    prompt = f"""
You are provided with diagram and the associated code that generated it. Based on criteria name,
the image has received a feedback that will help it to improve.
Refine the code to incorporate the following feedback

Existing Code:
{code}

Feedback from evaluation:
{feedback}

Return only improved Python code.
"""

    # --------------------------------------------------
    # Call LLM to generate refined code
    # --------------------------------------------------
    new_code_raw = call_llm(prompt, f"refine_{step}")

    # --------------------------------------------------
    # Extract executable Python code from the LLM output
    # --------------------------------------------------
    new_code = extract_python_code(new_code_raw)

    # --------------------------------------------------
    # Save refined code for logging and reproducibility
    # --------------------------------------------------
    save_code(f"refined_code_{step}", new_code)

    # --------------------------------------------------
    # Return improved code
    # --------------------------------------------------
    return new_code