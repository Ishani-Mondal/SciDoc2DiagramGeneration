from critics.completeness import completeness_critic
from critics.faithfulness import faithfulness_critic
from critics.layout import layout_critic
from generation.renderer import render_diagram
from refinement.refiner import refine_diagram
from config import MAX_ITER, TARGET_SCORE
from utils.logger import log


def refine_sum_maf(intent, retriever, code):
    """
    Refine a generated diagram using the Summarization-Based
    Multi-Aspect Feedback (Sum-MAF) loop.

    This function implements an iterative refinement strategy where an
    initial diagram is first rendered from code, then evaluated by
    three critic modules:

        1. Completeness critic
        2. Faithfulness critic
        3. Layout critic

    If the diagram does not meet the target quality threshold, the
    feedback from all critics is merged and used to revise the code.
    The revised code is then rendered again, and the process repeats
    until either:
        - all critic scores reach the target threshold, or
        - the maximum number of iterations is exhausted, or
        - rendering fails.

    Parameters
    ----------
    intent : str
        User-provided intent describing the desired diagram.
        Example:
            "Create a bar chart comparing precision of NASH with
             the three best baselines on Reuters."

    retriever : Retriever
        Multimodal retriever used by the critics to fetch relevant
        textual or visual evidence from the source document.

    code : str
        Executable Python code that generates the diagram.

    Returns
    -------
    str or None
        Path to the final rendered diagram image if successful.
        Returns the last successfully rendered diagram. If rendering
        fails early, the function may return None depending on the
        renderer behavior.

    Notes
    -----
    - The completeness and faithfulness critics use the retriever
      internally to gather evidence from the source document.
    - The layout critic evaluates only the rendered diagram image
      and the intent.
    - Feedback from all critics is concatenated into one list and
      passed to the refiner to update the code.
    """

    # --------------------------------------------------
    # Render the initial diagram from the provided code
    # --------------------------------------------------
    # The first draft is saved using the name "initial".
    diagram = render_diagram(code, "initial")

    # --------------------------------------------------
    # Run iterative multi-aspect refinement
    # --------------------------------------------------
    # The loop continues for at most MAX_ITER refinement rounds.
    for step in range(MAX_ITER):

        log(f"MAF iteration {step}")

        # --------------------------------------------------
        # Evaluate current diagram using all critic modules
        # --------------------------------------------------
        # Each critic returns:
        #   - a scalar score
        #   - a list of feedback items
        #
        # Completeness critic:
        #   checks whether all required content is present.
        c_score, c_fb = completeness_critic(intent, retriever, diagram, code)

        # Faithfulness critic:
        #   checks whether the content in the diagram is true to the paper.
        f_score, f_fb = faithfulness_critic(intent, retriever, diagram, code)

        # Layout critic:
        #   checks readability, structure, spacing, and visual quality.
        l_score, l_fb = layout_critic(intent, diagram)

        log(f"Scores C:{c_score} F:{f_score} L:{l_score}")

        # --------------------------------------------------
        # Early stopping condition
        # --------------------------------------------------
        # Stop refinement if all critic scores are above the target threshold.
        if min(c_score, f_score, l_score) >= TARGET_SCORE:
            break

        # --------------------------------------------------
        # Merge feedback from all critics
        # --------------------------------------------------
        # Sum-MAF combines all feedback together before refinement.
        feedback = c_fb + f_fb + l_fb

        # --------------------------------------------------
        # Refine the code using aggregated feedback
        # --------------------------------------------------
        # The refiner updates the diagram-generating Python code.
        code = refine_diagram(code, feedback, step)

        # --------------------------------------------------
        # Re-render the diagram after refinement
        # --------------------------------------------------
        diagram = render_diagram(code, f"refined_{step}")

        # --------------------------------------------------
        # Stop if rendering fails
        # --------------------------------------------------
        # This prevents the loop from continuing with an invalid diagram.
        if diagram is None:
            log("Diagram generation failed — stopping refinement")
            break

    # --------------------------------------------------
    # Return the final available diagram
    # --------------------------------------------------
    return diagram