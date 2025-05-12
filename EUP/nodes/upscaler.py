
#### Comfy UI Lib's ####
from nodes import MAX_RESOLUTION
import comfy.samplers
import comfy.sample

#### Services ####
from EUP.services.status import StatusService
from EUP.services.upscaler import PixelTiledKSampleUpscalerService, AdvancedPixelTiledKSampleUpscalerService
#from EUP.services.upscaler import ULtimate_16KANDUPUpscalerService

SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]']

class IterativeLatentUpscale:

    def __init__(self):
        self.statusService = StatusService()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "step_mode": (["simple", "geometric"], {"default": "simple"})
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent", "vae")
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, samples, upscale_factor, steps, temp_prefix, upscaler : PixelTiledKSampleUpscalerService, step_mode="simple", unique_id=None):
        w = samples['samples'].shape[3]*8  # image width
        h = samples['samples'].shape[2]*8  # image height

        if temp_prefix == "":
            temp_prefix = None

        if step_mode == "geometric":
            upscale_factor_unit = pow(upscale_factor, 1.0/steps)
        else:  # simple
            upscale_factor_unit = max(0, (upscale_factor - 1.0) / steps)

        current_latent = samples
        noise_mask = current_latent.get('noise_mask')
        scale = 1

        for i in range(steps-1):
            if step_mode == "geometric":
                scale *= upscale_factor_unit
            else:  # simple
                scale += upscale_factor_unit

            new_w = w*scale
            new_h = h*scale
            self.statusService.updateNodeStatus(unique_id, f"{i+1}/{steps} steps | x{scale:.2f}", (i+1)/steps)
            print(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
            step_info = i, steps
            current_latent = upscaler.upscaleShape(step_info, current_latent, new_w, new_h, temp_prefix)
            if noise_mask is not None:
                current_latent['noise_mask'] = noise_mask

        if scale < upscale_factor:
            new_w = w*upscale_factor
            new_h = h*upscale_factor
            self.statusService.updateNodeStatus(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            print(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps-1, steps
            current_latent = upscaler.upscaleShape(step_info, current_latent, new_w, new_h, temp_prefix)

        self.statusService.updateNodeStatus(unique_id, "", None)

        return current_latent, upscaler.vae

class PixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, denoise, upscale_model_opt=None, 
             pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        
        upscaler = PixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            sampler_advanced_pars=sampler_advanced_pars, denoise=denoise, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, 
            tile_size=max(512, 512), tile_cnet_strength=tile_cnet_strength, overlap=overlap 
        )
        
        return (upscaler, )


class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "tiling_strategy": (["simple", "random", "padded", "adjacency-padded", "context-padded", "overlaping", "adaptive", "hierarchical", "random-hierarchical", "non-uniform"],),
                    "tiling_strategy_pars": ("TILING_STRG_PARS",),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, sampler_advanced_pars, denoise, basic_pipe, upscale_model_opt=None, pk_hook_opt=None, 
             tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        
        model, _, vae, positive, negative = basic_pipe
        upscaler = PixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, 
            sampler_advanced_pars=sampler_advanced_pars, denoise=denoise, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, 
            tile_size=max(512, 512), tile_cnet_strength=tile_cnet_strength, overlap=overlap)
        
        return (upscaler, )
    

class AdvancedPixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                        },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, model, vae, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, start_at_step, end_at_step, 
             return_with_leftover_noise, upscale_model_opt=None, pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        
        upscaler = AdvancedPixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, add_noise=add_noise, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, 
            positive=positive, negative=negative, sampler_advanced_pars=sampler_advanced_pars, start_at_step=start_at_step, end_at_step=end_at_step, 
            return_with_leftover_noise=return_with_leftover_noise, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, 
            tile_size=max(512, 512), tile_cnet_strength=tile_cnet_strength, overlap=overlap 
        )
        
        return (upscaler, )
    
class AdvancedPixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "add_noise": (["enable", "disable"], ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "sampler_advanced_pars": ("SAMPLER_ADVANCED_PARS",),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "EUP - Ultimate Pack/Upscale"

    def doit(self, scale_method, add_noise, seed, steps, cfg, sampler_name, scheduler, sampler_advanced_pars, start_at_step, end_at_step, 
            return_with_leftover_noise, basic_pipe, upscale_model_opt=None, pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        
        model, _, vae, positive, negative = basic_pipe
        upscaler = AdvancedPixelTiledKSampleUpscalerService(
            scale_method=scale_method, model=model, vae=vae, add_noise=add_noise, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, 
            positive=positive, negative=negative, sampler_advanced_pars=sampler_advanced_pars, start_at_step=start_at_step, end_at_step=end_at_step, 
            return_with_leftover_noise=return_with_leftover_noise, upscale_model_opt=upscale_model_opt, hook_opt=pk_hook_opt, tile_cnet_opt=tile_cnet_opt, 
            tile_size=max(512, 512), tile_cnet_strength=tile_cnet_strength, overlap=overlap 
        )
        
        return (upscaler, )

'''
from custom_nodes.comfyui_ultimatesdupscale.nodes import MODES, SEAM_FIX_MODES

class ULtimate_16KANDUPUpscaler():

    def __init__(self):
        self.ultimateUpscale = ULtimate_16KANDUPUpscalerService()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image" : ("IMAGE",),
                # Sampling Params
                "model" : ("MODEL",),
                "positive" :("CONDITIONING",),
                "negative" : ("CONDITIONING",),
                "vae" :("VAE",),
                "upscale_by" : ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05}),
                "seed" :("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps" : ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "sampler_name" : (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler" : (comfy.samplers.KSampler.SCHEDULERS,),
                # Upscale Params
                "upscale_model" : ("UPSCALE_MODEL",),
                "pickle_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                "tile_num" :("INT", {"default": 2, "min": 0, "max": 0xffffffffffffffff}),
                "mode_type" : (list(MODES.keys()),),
                "mask_blur" :("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding" : ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                # Seam fix params
                "seam_fix_mode" : (list(SEAM_FIX_MODES.keys()),),
                "seam_fix_denoise" : ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width" : ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "seam_fix_mask_blur" : ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding" : ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                # Misc
                "force_uniform_tiles" : ("BOOLEAN", {"default": True}),
                "tiled_decode" : ("BOOLEAN", {"default": False}),
            },
                }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed, steps,
                sampler_name, scheduler, upscale_model, pickle_prefix, tile_num, mode_type, 
                mask_blur, tile_padding, seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, 
                custom_sampler=None, custom_sigmas=None):
        
        return self.ultimateUpscale.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae, upscale_by=upscale_by, seed=seed, steps=steps, sampler_name=sampler_name, scheduler=scheduler, 
            upscale_model=upscale_model, pickle_prefix=pickle_prefix, tile_num=tile_num, mode_type=mode_type, mask_blur=mask_blur, tile_padding=tile_padding, 
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise, seam_fix_mask_blur=seam_fix_mask_blur, seam_fix_width=seam_fix_width, 
            seam_fix_padding=seam_fix_padding, force_uniform_tiles=force_uniform_tiles, tiled_decode=tiled_decode, custom_sampler=custom_sampler, custom_sigmas=custom_sigmas
        )
'''
NODE_CLASS_MAPPINGS = {
    "EUP - Iterative Latent Upscaler": IterativeLatentUpscale,
    "EUP - Pixel TiledKSample Upscaler Provider": PixelTiledKSampleUpscalerProvider,
    "EUP - Pixel TiledKSample Upscaler Provider Pipe": PixelTiledKSampleUpscalerProviderPipe,
    "EUP - Advanced Pixel TiledKSample Upscaler Provider": AdvancedPixelTiledKSampleUpscalerProvider,
    "EUP - Advanced Pixel TiledKSample Upscaler Provider Pipe": AdvancedPixelTiledKSampleUpscalerProviderPipe,
    #"EUP - Ultimate 16K and Up Tiling": ULtimate_16KANDUPUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}

