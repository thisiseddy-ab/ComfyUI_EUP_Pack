#### Comfy Lib's #####
import comfy.controlnet
import comfy.sampler_helpers

#### Third Party Lib's #####

#### Services #####
from EUP.services.tensor import TensorService

import copy

class Spatial_Condition_Service():
    def __init__(self):
        self.tensorService = TensorService()

    def extractSpatialConds(self, positive, negative, shape, device):
        spatial_conds_pos = [
            (c[1]['area'] if 'area' in c[1] else None, 
             comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in positive
        ]
        spatial_conds_neg = [
            (c[1]['area'] if 'area' in c[1] else None, 
             comfy.sampler_helpers.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
            for c in negative
        ]
        return (spatial_conds_pos, spatial_conds_neg)

    def sliceSpatialConds(self, tileSize, cond, area):
        x, y, tile_w, tile_h = tileSize
        tile_h_end = y + tile_h
        tile_w_end = x + tile_w
        coords = area[0]
        mask = area[1]

        # Deepcopy to avoid mutating shared reference
        cond = (cond[0], cond[1].copy())

        # Handle area slicing
        if coords is not None:
            h_len, w_len, h, w = coords
            h_end = h + h_len
            w_end = w + w_len
            if h < tile_h_end and h_end > y and w < tile_w_end and w_end > x:
                new_h = max(0, h - y)
                new_w = max(0, w - x)
                new_h_end = min(tile_h_end, h_end - y)
                new_w_end = min(tile_w_end, w_end - x)
                cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
            else:
                return (cond, True)

        # Handle mask slicing
        if mask is not None:
            new_mask = self.tensorService.getSlice(mask, (x, y, tile_w, tile_h))
            if new_mask.sum().cpu() == 0.0:
                return (cond, True)
            else:
                cond[1]['mask'] = new_mask

        return (cond, False)

    def prepareSlicedConds(self, pos, neg, spatial_conds_pos, spatial_conds_neg, tileSize):
        x, y, tile_w, tile_h = tileSize

        pos = [self.sliceSpatialConds((x, y, tile_w, tile_h), copy.deepcopy(c), area)
               for c, area in zip(pos, spatial_conds_pos)]
        pos = [c for c, ignore in pos if not ignore]  # Ignore invalid conditions

        neg = [self.sliceSpatialConds((x, y, tile_w, tile_h), copy.deepcopy(c), area)
               for c, area in zip(neg, spatial_conds_neg)]
        neg = [c for c, ignore in neg if not ignore]  # Ignore invalid conditions

        return (pos, neg)

class ConditionService():
    def __init__(self):
        self.sptcondService = Spatial_Condition_Service()