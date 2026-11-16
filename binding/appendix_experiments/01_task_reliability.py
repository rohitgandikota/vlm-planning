"""How reliable is the task with no patch? (Appendix A.7, first paragraph)

Before any patch, we ask about every shape in every image of the candidate pairs: four questions
per image, two images per pair. We read the answer in three ways.
    next token     the next token after "Answer:" is the letter (case is ignored)
    free           with nothing after the assistant tag, the model writes up to 6 tokens; after
                   removing spaces and punctuation the text is the letter (case is ignored)
    gaze           the published Gaze Heads look at the asked panel from the colon token

The candidates are the pairs that `one_answer_pairs` draws before its no-patch filter. The paper
asks about the first 904 candidates, which is every pair it drew (7,232 questions). Intervals
resample whole candidate pairs.

    python binding/appendix_experiments/01_task_reliability.py --num_pairs 904

Paper: next token right in 98.9%, free generation in 98.9%, Gaze Heads on the asked panel in 97.9%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_image, one_answer_question, sample_one_answer_pair
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import run_model, write_tokens

PUNCTUATION = ".,:;!?\"'"


def ask_every_shape(model, side):
    """Ask about each of the four shapes of one image. Yields (asked panel, letter, three answers)."""
    image = one_answer_image(side)
    for panel, (shape, letter) in enumerate(zip(side["shapes"], side["letters"])):
        run = run_model(model, image, one_answer_question(shape))
        looked_at = max(range(4), key=run.panel_attention(HEADS["published_gaze"], "last").__getitem__)
        free = write_tokens(model, run_model(model, image, one_answer_question(shape), prefill="", record=False),
                            max_tokens=6)
        yield panel, letter, run.answer, free, looked_at


def main():
    args = parse_args(__doc__, num_pairs=904)
    model = load_model(args.model, args.device)
    results = Results("task_reliability", args)
    for candidate in range(args.start, args.start + args.num_pairs):
        pair = sample_one_answer_pair(candidate)
        for side in (pair["source"], pair["target"]):
            for panel, letter, answer, free, looked_at in ask_every_shape(model, side):
                results.add(candidate, next_token=answer.strip().upper() == letter,
                            free=free.strip().strip(PUNCTUATION).strip().upper() == letter,
                            gaze=looked_at == panel, answer=answer, free_text=free)
    results.report("next_token", paper=98.9, label="next token after 'Answer:' is right")
    results.report("free", paper=98.9, label="free generation is right")
    results.report("gaze", paper=97.9, label="Gaze Heads look at the asked panel")
    results.save()


if __name__ == "__main__":
    main()
