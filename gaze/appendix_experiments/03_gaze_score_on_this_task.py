"""The Gaze score measured on our task, and three rankings compared (Appendix F.2, Figure 9a).

The published Gaze Heads were ranked by where heads look while the model describes comic strips.
We measure the same score on the one-answer task, with no patch: for every image of a pair and
every shape in it, we ask for the letter inside that shape, and take the attention each head puts
from the colon token on the image tokens of the asked panel (the raw weights, summed over the
panel). The score of a head is its mean over all questions. Then we count the heads shared by the
top K of three rankings: our query-patch search (HEADS["gaze_ranking"]), this score, and the
published ranking.

    python gaze/appendix_experiments/03_gaze_score_on_this_task.py --num_pairs 500

--num_pairs is the number of pairs; each gives 2 images x 4 shapes = 8 questions.

Paper: the published and the re-measured ranking share 57 of their top 100 heads. In the top 32,
our ranking shares 20 heads with the re-measured score and 10 with the published ranking. 23 of the
32 gaze heads are in the published top 100, 29 in the re-measured top 100. The paper's score came
from an earlier set of 250 pairs (2,000 questions); its top 100 is HEADS["gaze_score_on_this_task"].
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository root

from utils.dataset import one_answer_image, one_answer_pairs, one_answer_question
from utils.experiment import Results, parse_args
from utils.heads import HEADS, rank_heads
from utils.model import load_model, n_heads, n_layers
from utils.patching import Patch, run_model, run_patches


def main():
    args = parse_args(__doc__)
    model = load_model(args.model, args.device)
    results = Results("gaze_score_on_this_task", args)
    on_panel, questions = torch.zeros(n_layers(model), n_heads(model)), 0
    for pair in one_answer_pairs(model, args.split, args.start + args.num_pairs)[args.start:]:
        for side in (pair["source"], pair["target"]):
            for panel, shape in enumerate(side["shapes"]):
                run = run_model(model, one_answer_image(side), one_answer_question(shape), record=False)
                attention = run_patches(model, run, [Patch()], keep_attention=True)["attention"]    # the colon token
                positions = run.panel_positions(panel)
                for layer in range(n_layers(model)):
                    on_panel[layer] += attention[layer][0][:, positions].sum(dim=1).float().cpu()
                questions += 1
        print(f"pair {pair['index']} measured", flush=True)
    measured = rank_heads(on_panel / questions)
    rankings = {"ours": HEADS["gaze_ranking"], "measured": measured, "published": HEADS["published_gaze_ranking"]}
    shared = lambda a, b, k: len(set(rankings[a][:k]) & set(rankings[b][:k]))
    for k in (8, 16, 32, 64, 128, 256):
        print(f"top {k:3d}: ours and measured share {shared('ours', 'measured', k):3d}, ours and published "
              f"{shared('ours', 'published', k):3d}, measured and published {shared('measured', 'published', k):3d}")
    print(f"published and measured share {shared('measured', 'published', 100)} of their top 100 (paper: 57)")
    print(f"gaze heads in the measured top 100: {len(set(HEADS['gaze']) & set(measured[:100]))} (paper: 29)")
    print(f"gaze heads in the published top 100: {len(set(HEADS['gaze']) & set(rankings['published'][:100]))} (paper: 23)")
    print(f"top 100 shared with the paper's measured ranking: {len(set(measured[:100]) & set(HEADS['gaze_score_on_this_task']))}")
    results.save(questions=questions, ranking=measured, scores=(on_panel / questions).tolist())


if __name__ == "__main__":
    main()
