#### Comfy Lib's #####
import comfy.controlnet

#### Third Party Lib's #####
import torch

#### Services #####
from EUP.services.tensor import TensorService
from EUP.services.list import ListService

class T2IService():

    def __init__(self):
        self.tensorService = TensorService()
        self.listService = ListService()

    def extract_T2I(self, positive, negative):
        t2is = [c['control'] for (_, c) in positive + negative if 'control' in c]
        t2is = [x for m in t2is for x in self.listService.recursion_toList(m, "previous_controlnet")] 
        return [x for x in t2is if isinstance(x, comfy.controlnet.T2IAdapter)]
    
    def prepareT2I_imgs(self, T2Is, shape):
        return [
            torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
            if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
            for m in T2Is
        ]
    
    def slices_T2I(self, tileSize, model:comfy.controlnet.ControlBase, img):
        x, y, tile_w, tile_h = tileSize
        model.control_input = None
        if img is None:
            img = model.cond_hint_original
        
        # Ensure slicing logic works for the tile
        hint = self.tensorService.getSlice(img, (x * 8, y * 8, tile_w * 8, tile_h * 8))
        
        if isinstance(model, comfy.controlnet.ControlLora):
            model.cond_hint = hint.float().to(model.device)
        else:
            model.cond_hint = hint.to(model.control_model.dtype).to(model.control_model.device)
            

    def prepareSlicedT2Is(self, neg, T2Is, T2I_imgs, tileSize):
        x, y, tile_w, tile_h = tileSize
        for m, img in zip(T2Is, T2I_imgs):
            self.slices_T2I((x, y, tile_w, tile_h), m, img)
