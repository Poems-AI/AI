import torch


__all__ = ['masked_min', 'get_positions_between']


def masked_min(t, mask, axis=None, inf_val=1000000000):
    return (t + ~mask * inf_val).min(axis=axis).values


def get_positions_between(t, ini_value, end_value):
    positions = torch.arange(0, t.shape[0])

    ini_mask = t == ini_value
    end_mask = t == end_value
    ini_positions = positions[ini_mask]
    end_positions = positions[end_mask]
    inf_val = 1000000000
    slice_end_positions = masked_min(end_positions[..., None].repeat(1, ini_positions.shape[0]),
                                     end_positions[..., None] > ini_positions,
                                     axis=0)
    slice_begin_positions = ini_positions
    slice_end_positions = slice_end_positions
    return slice_begin_positions, slice_end_positions
