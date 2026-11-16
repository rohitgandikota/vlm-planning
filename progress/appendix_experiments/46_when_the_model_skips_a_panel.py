"""When the model skips a panel, the last pointer is still right (Appendix E.2).

No patch. The model makes few errors on six panels, so we let it write whole lists on strips of ten
panels with seven matching panels (utils.dataset `long_strips`). After each written letter it
chooses between a comma and the end of its turn; if it goes on, we read its next letter after the
comma. Each answer is named (`label_list_answer`): correct, skip (a later matching panel), filler,
back, repeat, extra. At every answer that follows a written letter we read, with no patch:
    binding   the panel on which the 16 binding heads put most attention at the last answer token
    cursor    a ridge probe on the comma token's residual at layer 21 for the last written panel,
              fitted on the correct answers of the strips of the other parity (even or odd index)
    gaze      the panel on which the 32 gaze heads put most attention at the comma token
and compare the skips with correct answers at the same position in the list.

    python progress/appendix_experiments/46_when_the_model_skips_a_panel.py --num_pairs 400

--num_pairs is the number of strips. Paper: 75.8% of the 400 lists have no error. 3.2% of answers
are skips, and 86.5% of skips pass over a single matching panel. 1.3% of answers come from a panel
without the shape, 0.5% go back, 0.1% repeat a letter, and the model never stops early. At a skip:
binding heads on the written panel 100.0% (as in correct answers), cursor probe 92.1% against
97.5%, gaze heads on the correct next panel 25.4% against 95.6%, and on the panel the model then
writes 68.3%. A panel without the shape lies between the last written panel and the correct next
one in 71.4% of skips, against 28.2% of correct answers. This was an earlier, smaller study: the
paper's numbers come from these 400 strips, not from the 500 held-out pairs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import LETTERS, label_list_answer, last_written_panel, long_strips, next_matching_panel, run_list
from utils.experiment import Results, parse_args
from utils.heads import HEADS
from utils.model import load_model
from utils.patching import fit_linear_probe

PROBE_LAYER = 21


def write_list(model, strip):
    """Let the model write its whole list. Returns its answer steps and its stop choices, each named
    "correct", "early" (it stops with matching panels left) or "late" (it goes on with none left)."""
    written, steps, stop_choices = [], [], []
    for position in range(len(strip["matching"]) + 3):
        if written:                                                # first the choice: a comma, or the end of the turn
            stops = run_list(model, strip, written, end_on_letter=True, record=False).answer == "<end>"
            finished = next_matching_panel(strip, written) is None
            stop_choices.append("correct" if stops == finished else "early" if stops else "late")
            if stops:
                break
        run = run_list(model, strip, written)
        letter = run.answer.strip()
        step = {"position": position, "written": list(written), "letter": letter,
                "label": label_list_answer(strip, written, letter), "cursor": last_written_panel(strip, written),
                "target": next_matching_panel(strip, written),
                "answer_panel": strip["letters"].index(letter) if letter in strip["letters"] else -1,
                "gaze": int(np.argmax(run.panel_attention(HEADS["gaze"], "last"))),
                "residual": run.residual(PROBE_LAYER, "last").half().cpu().numpy()}
        if written:
            step["binding"] = int(np.argmax(run.panel_attention(HEADS["binding"], "last_answer")))
        steps.append(step)
        if len(letter) != 1 or letter not in LETTERS:
            break
        written.append(letter)
    return steps, stop_choices


def main():
    args = parse_args(__doc__, num_pairs=400)
    model = load_model(args.model, args.device)
    results = Results("when_the_model_skips_a_panel", args)
    strips = long_strips(args.start + args.num_pairs)[args.start:]
    steps = []
    for strip in strips:
        own, stop_choices = write_list(model, strip)
        results.add(strip["index"], list_without_error=all(s["label"] == "correct" for s in own) and set(stop_choices) <= {"correct"})
        for choice in stop_choices:
            results.add(strip["index"], early_stop=choice == "early", late_stop=choice == "late")
        for s in own:
            s["strip"] = strip
            results.add(strip["index"], **{f"answer_is_{name}": s["label"] == name for name in ("skip", "filler", "back", "repeat")})
            if s["label"] == "skip":
                passed = strip["matching"].index(s["answer_panel"]) - strip["matching"].index(s["target"])
                results.add(strip["index"], skip_by_one_panel=passed == 1)
        steps += own
    # The cursor probe: "which panel was written last", cross-fitted by the parity of the strip index.
    after_letter = [s for s in steps if s["cursor"] >= 0]
    n_panels = len(strips[0]["letters"])
    for parity in (0, 1):
        fit = [s for s in after_letter if s["strip"]["index"] % 2 != parity and s["label"] == "correct"]
        probe = fit_linear_probe([s["residual"] for s in fit], np.eye(n_panels)[[s["cursor"] for s in fit]])
        for s in [s for s in after_letter if s["strip"]["index"] % 2 == parity]:
            s["probe"] = int(np.argmax(probe([s["residual"]])[0]))
    # Skips after a written letter, against correct answers at the same positions in the list.
    skip_positions = {s["position"] for s in after_letter if s["label"] == "skip"}
    for s in after_letter:
        kind = s["label"] if s["label"] == "skip" else "correct" if s["label"] == "correct" and s["position"] in skip_positions else None
        if kind is None:
            continue
        p, t = s["cursor"], s["target"]
        between = any(k not in s["strip"]["matching"] for k in range(p + 1, t))     # a panel without the shape before t
        results.add(s["strip"]["index"], **{f"{kind}_binding_on_written_panel": s["binding"] == p,
                                            f"{kind}_probe_reads_written_panel": s["probe"] == p,
                                            f"{kind}_gaze_on_correct_next_panel": s["gaze"] == t,
                                            f"{kind}_gaze_on_answered_panel": s["gaze"] == s["answer_panel"],
                                            f"{kind}_panel_without_shape_between": between})
    results.report("list_without_error", paper=75.8, label="lists with no error")
    for name, paper in (("skip", 3.2), ("filler", 1.3), ("back", 0.5), ("repeat", 0.1)):
        results.report(f"answer_is_{name}", paper=paper, label=f"answers that are a {name}")
    results.report("skip_by_one_panel", paper=86.5, label="skips that pass over one matching panel")
    results.report("early_stop", paper=0.0, label="stop choices that stop early")
    results.report("late_stop", label="stop choices that go on with nothing left")
    for name, paper in (("binding_on_written_panel", (100.0, 100.0)), ("probe_reads_written_panel", (92.1, 97.5)),
                        ("gaze_on_correct_next_panel", (25.4, 95.6)), ("gaze_on_answered_panel", (68.3, None)),
                        ("panel_without_shape_between", (71.4, 28.2))):
        results.report(f"skip_{name}", paper=paper[0], label=f"at a skip: {name.replace('_', ' ')}")
        results.report(f"correct_{name}", paper=paper[1], label=f"at a correct answer: {name.replace('_', ' ')}")
    results.save()


if __name__ == "__main__":
    main()
