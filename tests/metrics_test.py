import math
from poemsai.metrics import compute_lm_metrics
import torch


def test_compute_lm_metrics():
    # (bs, seq length, vocab size) = (2, 3, 4)
    logits = torch.Tensor([
        [[0, 0.1, 0.2, 1.3], [0.1, 0, 0, 0], [-0.5, -0.2, -0.6, -0.9], [0.1, 0.2, 0.3, 0.4]],
        [[-0.1, 0.2, 0.3, 0.25], [1., 1.1, 0.7, 0.6], [0.5, 0.5, 0.5, 0.55], [0.1, 0.2, 0.3, 0.4]]
    ])
    preds = torch.Tensor([[3, 0, 1, 3], [2, 1, 3, 3]])
    labels_eq = torch.Tensor([[0, 3, 0, 1], [0, 2, 1, 3]])
    labels_dif = torch.Tensor([[0, 2, 1, 0], [0, 3, 0, 2]])
    labels_half = torch.Tensor([[0, 2, 0, 3], [0, 2, 2, 3]])
    metric = 'accuracy'
    
    assert math.isclose(compute_lm_metrics((logits, labels_eq))[metric], 1)
    assert math.isclose(compute_lm_metrics((logits, labels_dif))[metric], 0)
    assert math.isclose(compute_lm_metrics((logits, labels_half))[metric], 0.5)
    assert math.isclose(compute_lm_metrics((preds, labels_eq), expect_preds=True)[metric], 1)
    assert math.isclose(compute_lm_metrics((preds, labels_dif), expect_preds=True)[metric], 0)
    assert math.isclose(compute_lm_metrics((preds, labels_half), expect_preds=True)[metric], 0.5)
