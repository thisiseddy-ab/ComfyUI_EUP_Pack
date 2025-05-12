import copy 

#### Comfy Lib's #####
import comfy.controlnet

#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService
from EUP.services.list import ListService

class ControlNetService():

    def __init__(self):
        self.tensorService = TensorService()
        self.listService = ListService()

    def extractCnet(self, positive, negative):
        cnets = [c['control'] for (_, c) in positive + negative if 'control' in c]
        cnets = list(set([x for m in cnets for x in self.listService.recursion_toList(m, "previous_controlnet")]))  # Recursive extraction
        return [x for x in cnets if isinstance(x, comfy.controlnet.ControlNet)]

    def prepareCnet_imgs(self, cnets, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
            for m in cnets
        ]
    
    def sliceCnet(self, tileSize, model: comfy.controlnet.ControlBase, img):
        x, y, tile_w, tile_h = tileSize
        if img is None:
            img = model.cond_hint_original

        # Ensure slicing logic works for the tile
        hint = self.tensorService.getSlice(img, (x * 8, y * 8, tile_w * 8, tile_h * 8))
        
        if isinstance(model, comfy.controlnet.ControlLora):
            model.cond_hint = hint.float().to(model.device)
        else:
            model.cond_hint = hint.to(model.control_model.dtype).to(model.control_model.device)
        
    def prepareSlicedCnets(self, cnets, cnet_imgs, tileSize):
        x, y, tile_w, tile_h = tileSize
        for m, img in zip(cnets, cnet_imgs):
            self.sliceCnet((x, y, tile_w, tile_h), m, img)
