import numpy as np

from .renderer import recon_loss, soft_grid_mask


def mdl_cost(floors, repeats, beta_depth=1.0):
    rules = floors + 1
    depth = 2
    return rules + beta_depth * depth


def score_candidate(height, width, floors, repeats, feat_map, lam_rec, lam_mdl, beta_depth):
    mask = soft_grid_mask(height, width, floors, repeats)
    loss_rec = recon_loss(feat_map, mask)
    loss_mdl = mdl_cost(floors, repeats, beta_depth)
    return lam_rec * loss_rec + lam_mdl * loss_mdl


def search_best(height, width, floors_list, img_gray, feat_map, lam_rec, lam_mdl, beta_depth, rmin, rmax, beam_width=5):
    candidates = []
    for floors, bands in floors_list:
        for repeats in range(rmin, rmax + 1):
            score = score_candidate(height, width, floors, repeats, feat_map, lam_rec, lam_mdl, beta_depth)
            candidates.append((score, floors, repeats))
    candidates.sort(key=lambda item: item[0])
    return candidates[:beam_width]
