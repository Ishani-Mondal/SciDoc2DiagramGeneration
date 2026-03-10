
from critics.completeness import completeness_critic
from critics.faithfulness import faithfulness_critic
from critics.layout import layout_critic
from generation.renderer import render_diagram
from refinement.refiner import refine_diagram
from config import MAX_ITER, TARGET_SCORE
from utils.logger import log


def refine_seq_maf(intent, retriever, code):
    """
    Refine a generated diagram using Sequential Multi-Aspect Feedback (Seq-MAF).

    Unlike Sum-MAF, which aggregates feedback from all critics in each iteration,
    Seq-MAF refines the diagram critic-by-critic in a fixed order:

        1. Completeness
        2. Faithfulness
        3. Layout

    For each critic:
        - evaluate the current diagram
        - if the score is below TARGET_SCORE, refine the code using only that critic's feedback
        - re-render the diagram
        - repeat until the critic is satisfied or the iteration budget is exhausted

    Parameters
    ----------
    intent : str
        User intent describing the target diagram.

    retriever : Retriever
        Multimodal retriever used by critics to fetch supporting evidence.

    code : str
        Python code that generates the diagram.

    Returns
    -------
    str or None
        Path to the final rendered diagram, or None if rendering fails.
    """

    # --------------------------------------------------
    # Render initial diagram
    # --------------------------------------------------
    diagram = render_diagram(code, "initial")

    if diagram is None:
        log("Initial diagram generation failed.")
        return None

    # --------------------------------------------------
    # Critic schedule for sequential refinement
    # --------------------------------------------------
    critics = [
        ("completeness", lambda d, c: completeness_critic(intent, retriever, d, c)),
        ("faithfulness", lambda d, c: faithfulness_critic(intent, retriever, d, c)),
        ("layout", lambda d, c: layout_critic(intent, d)),
    ]

    global_step = 0

    # --------------------------------------------------
    # Sequentially satisfy each critic
    # --------------------------------------------------
    for critic_name, critic_fn in critics:

        log(f"Starting Seq-MAF stage: {critic_name}")

        for local_step in range(MAX_ITER):

            log(f"Seq-MAF [{critic_name}] iteration {local_step}")

            # Evaluate only the current critic
            score, feedback = critic_fn(diagram, code)

            log(f"{critic_name} score: {score}")

            # If the current critic is satisfied, move to next critic
            if score >= TARGET_SCORE:
                log(f"{critic_name} satisfied with score {score}")
                break

            # Refine using only this critic's feedback
            code = refine_diagram(code, feedback, f"{critic_name}_{global_step}")

            # Re-render updated diagram
            diagram = render_diagram(code, f"seq_{critic_name}_{global_step}")

            if diagram is None:
                log(f"Diagram generation failed during {critic_name} refinement")
                return None

            global_step += 1

        else:
            log(f"Max iterations reached for {critic_name}; moving to next critic")

    # --------------------------------------------------
    # Final logging with all critic scores
    # --------------------------------------------------
    try:
        c_score, _ = completeness_critic(intent, retriever, diagram, code)
        f_score, _ = faithfulness_critic(intent, retriever, diagram, code)
        l_score, _ = layout_critic(intent, diagram)

        log(f"Final Seq-MAF scores -> C:{c_score} F:{f_score} L:{l_score}")
    except Exception as e:
        log(f"Warning: final Seq-MAF scoring failed: {e}")

    return diagram