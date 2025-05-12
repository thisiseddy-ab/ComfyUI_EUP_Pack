#### Comfy Lib's ####
import comfy.model_management
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils

#### Third Party Lib's #####
import latent_preview
from tqdm.auto import tqdm

#### My Services's #####
from EUP.services.latent import LatentTilerService, LatentMergerService

#### Custom Nodes ####
from EUP.nodes.latent import LatentMerger

import torch

class KSamplerService:
    def __init__(self):
        self.latentService = LatentTilerService()
    
    def commonKsampler(self, model, seed, steps, actual_steps, cfg, sampler_name, scheduler, positive, negative, latent, sampler_advanced_pars, denoise=1.0, disable_noise=False, 
                       start_step=None, last_step=None, force_full_denoise=False,
                       ):
        ksampler_type, tiling_strategy, tiling_strategy_pars, noise_strategy, noise_strategy_pars, noise_mask_strategy, noise_mask_strategy_pars, denoise_strategy, denoise_strategy_pars, blend_strategy, blend_strategy_pars = sampler_advanced_pars

        device = comfy.model_management.get_torch_device()

        # Step 1: Load Additional Models
        conds0 = {
        "positive": comfy.sampler_helpers.convert_cond(positive),
        "negative": comfy.sampler_helpers.convert_cond(negative)
        }

        conds = {}
        for k in conds0:
            conds[k] = list(map(lambda a: a.copy(), conds0[k]))

        modelPatches, inference_memory = comfy.sampler_helpers.get_additional_models(conds, model.model_dtype())

        comfy.model_management.load_models_gpu([model] + modelPatches, model.memory_required(latent.get("samples").shape) + inference_memory)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

        # Step 2: Tile Latent **and Noise**
        tiled_latent = self.latentService.tileLatent(
            model=model, sampler=sampler, seed=seed, actual_steps=actual_steps, latent_image=latent, positive=positive, negative=negative, sampler_advanced_pars=sampler_advanced_pars, 
            disable_noise=disable_noise, start_step=start_step,
        )

        # Step 3: Prepare Needet Resources
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if not disable_pbar:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)
        
        total_tiles = sum(len(tile_group) for tile_pass in tiled_latent for tile_group in tile_pass)
        total_steps = steps * total_tiles
        
        # Step 4: Sample Tiles
        with tqdm(total=total_steps) as pbar_tqdm:
            proc_tiled_latent = self.sampleTiles(ksampler_type, sampler, tiled_latent, steps, seed, total_steps, total_tiles, pbar_tqdm, disable_pbar, 
                                                    previewer, cfg, start_step, last_step, force_full_denoise)
        # Step 5: Merge Tiles
        merger = LatentMerger()
        merged_latent = merger.mergeTiles(proc_tiled_latent)
        
        comfy.sampler_helpers.cleanup_additional_models(modelPatches)
        
        # Step 5: Return Merge Tiles
        #out = latent.copy()
        #out["samples"] = merged_latent[0].get("samples").to("cpu")
        #return (out, ),
        return merged_latent

    

    def show_tile(self, tensor, title="Image"):
        import matplotlib.pyplot as plt
        import numpy as np
        from torchvision import transforms
        # Convert tensor to numpy array for display
        image = tensor.squeeze(0).cpu().detach().numpy()  # Assuming tensor has shape (B, C, H, W)
        
        # Normalize to [0, 1] for visualization
        image = np.clip(image, 0, 1)  # Clip to [0, 1] if needed

        # Display with matplotlib
        plt.imshow(image.transpose(1, 2, 0))  # Transpose to (H, W, C) for display
        plt.title(title)
        plt.axis('off')
        plt.show()


    def sampleTiles(self, ksampler_type, sampler: comfy.samplers.KSampler, tiled_latent, steps, seed, total_steps, total_tiles, pbar_tqdm, disable_pbar, previewer, 
                    cfg, start_step, last_step, force_full_denoise):
        
        pbar = comfy.utils.ProgressBar(steps)
        proc_tiled_latent = []
        global_tile_index = 0  # Track tiles across all passes

        for tile_pass in tiled_latent:
            tile_pass_l = []
            for i, tile_group in enumerate(tile_pass):
                tile_group_l = []
                for (tile, tile_position, noise_tile, denoise_mask, blend_mask, mp, mn, tiled_steps) in tile_group:
                    tile_index = global_tile_index 

                    if  tile_position is not None:
                        tile.to("cpu")
                    if noise_tile is not None:
                        noise_tile.to("cpu")
                    if(denoise_mask is not None):
                        denoise_mask.to("cpu")
                    if(blend_mask is not None):
                        blend_mask.to("cpu")

                    denoise_mask = None

                    if ksampler_type == "Eddy's":
                        processed_tile = sampler.sample(
                            noise=noise_tile,  
                            positive=mp, 
                            negative=mn, 
                            cfg=cfg, 
                            latent_image=tile,  
                            start_step=start_step + i * tiled_steps,
                            last_step=start_step + i*tiled_steps + tiled_steps,
                            force_full_denoise=force_full_denoise and i+1 == last_step - start_step,
                            denoise_mask=denoise_mask,
                            callback=lambda step, x0, x, total_steps=total_steps, tile_index=tile_index: self.updateProgress(step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar),
                            disable_pbar=disable_pbar, 
                            seed=seed
                        )
                    elif ksampler_type == "BlenderNeko's":
                        processed_tile = sampler.sample(
                            noise=noise_tile,  
                            positive=mp, 
                            negative=mn, 
                            cfg=cfg, 
                            latent_image=tile,  
                            start_step=start_step,
                            last_step=last_step,
                            force_full_denoise=force_full_denoise,
                            denoise_mask=denoise_mask,
                            callback=lambda step, x0, x, total_steps=total_steps, tile_index=tile_index: self.updateProgress(step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar),
                            disable_pbar=disable_pbar, 
                            seed=seed
                    )

                    tile_group_l.append((processed_tile.to("cpu"), tile_position, blend_mask))
                    global_tile_index += 1  # Increment index

                tile_pass_l.append(tile_group_l)
            proc_tiled_latent.append(tile_pass_l)

        return proc_tiled_latent
    
    def updateProgress(self, step, x0, x, total_steps, tile_index, total_tiles, pbar_tqdm, disable_pbar, previewer, pbar):
        # Update the overall progress bar
        pbar_tqdm.update(1)

        # Update tile-specific progress
        if not disable_pbar:
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)

            pbar_tqdm.set_description(f"[EUP - Tiled KSampler]")

            # Update the postfix with correct total and tile progress
            postfix = {
                "Tile Progress": f"Tile {tile_index + 1}/{total_tiles}" 
            }
            pbar_tqdm.set_postfix(postfix)

            # Update the progress bar with the preview image (optional)
            pbar.update_absolute(step, preview=preview_bytes)  