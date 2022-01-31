import math
from poemsai.metrics import compute_lm_metrics, MetadataLessLoss
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


def test_metadataless_loss():
    class FakeLoss:
        def __init__(self):
            self.call_args = []
            
        def __call__(self, preds, target):
            self.call_args.append((preds, target))
            
    BOV_ID = 97
    EOV_ID = 98
    EOP_ID = 99
    IGN_IDX = -1
    loss_bov_tag = MetadataLessLoss(FakeLoss(), begin_verse_id=BOV_ID, end_verse_id=None, 
                                    end_poem_id=None, ignore_index=IGN_IDX)
    loss_end_tags = MetadataLessLoss(FakeLoss(), begin_verse_id=None, end_verse_id=EOV_ID, 
                                     end_poem_id=EOP_ID, ignore_index=IGN_IDX)
    loss_all_tags = MetadataLessLoss(FakeLoss(), begin_verse_id=BOV_ID, end_verse_id=EOV_ID, 
                                     end_poem_id=EOP_ID, ignore_index=IGN_IDX)
    labels = torch.Tensor([
        [BOV_ID, 1, 2, EOV_ID, BOV_ID, 3, EOV_ID, EOP_ID, BOV_ID, 4, 5, EOV_ID, BOV_ID, 6, EOV_ID, EOP_ID, BOV_ID, 7, 8, 9, EOV_ID],
        [BOV_ID, 1, 2, EOV_ID, 24, 35, BOV_ID, 3, EOV_ID, 28, EOP_ID, 23, 34, 56, BOV_ID, 4, EOV_ID, 77, 88, BOV_ID, 5],
        [30, 40, BOV_ID, 1, EOV_ID, 33, BOV_ID, 2, 3, EOV_ID, 44, 55, EOP_ID, 45, BOV_ID, 4, EOV_ID, EOP_ID, 88, 56, 62],
        [1, 2, EOV_ID, 3, EOP_ID, 23, 43, BOV_ID, 4, 5, 6, EOV_ID, BOV_ID, 7, 8, EOV_ID, 12, 15, 14, 18, 19]
    ])
    preds = torch.rand(*labels.shape, 5)
    loss_bov_tag(preds, labels)
    loss_end_tags(preds, labels)
    loss_all_tags(preds, labels)
    
    expected_loss_bov_target = labels
    expected_loss_end_tags_target = torch.Tensor([
        [BOV_ID, 1, 2, EOV_ID, BOV_ID, 3, EOV_ID, IGN_IDX, BOV_ID, 4, 5, EOV_ID, BOV_ID, 6, EOV_ID, IGN_IDX, BOV_ID, 7, 8, 9, EOV_ID],
        [BOV_ID, 1, 2, EOV_ID, 24, 35, BOV_ID, 3, EOV_ID, IGN_IDX, IGN_IDX, 23, 34, 56, BOV_ID, 4, EOV_ID, 77, 88, BOV_ID, 5],
        [30, 40, BOV_ID, 1, EOV_ID, 33, BOV_ID, 2, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 45, BOV_ID, 4, EOV_ID, IGN_IDX, 88, 56, 62],
        # In this case, the end of this sequence is ambiguous, you can't know if the final tokens are between EOV and EOP 
        # or between two EOV
        [1, 2, EOV_ID, IGN_IDX, IGN_IDX, 23, 43, BOV_ID, 4, 5, 6, EOV_ID, BOV_ID, 7, 8, EOV_ID, 12, 15, 14, 18, 19]
    ])
    expected_loss_all_tags_target = torch.Tensor([
        [IGN_IDX, 1, 2, EOV_ID, IGN_IDX, 3, EOV_ID, IGN_IDX, IGN_IDX, 4, 5, EOV_ID, IGN_IDX, 6, EOV_ID, IGN_IDX, IGN_IDX, 7, 8, 9, EOV_ID],
        [IGN_IDX, 1, 2, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, 5],
        [IGN_IDX, IGN_IDX, IGN_IDX, 1, EOV_ID, IGN_IDX, IGN_IDX, 2, 3, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX],
        [1, 2, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, 4, 5, 6, EOV_ID, IGN_IDX, 7, 8, EOV_ID, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX, IGN_IDX]
    ])    
    
    assert loss_bov_tag.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_bov_tag.inner_loss.call_args[0][1] == expected_loss_bov_target)
    assert loss_end_tags.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_end_tags.inner_loss.call_args[0][1] == expected_loss_end_tags_target)
    assert loss_all_tags.inner_loss.call_args[0][0] is preds
    assert torch.all(loss_all_tags.inner_loss.call_args[0][1] == expected_loss_all_tags_target)
