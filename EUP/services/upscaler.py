#### BuiltIn Libs ####
import os

#### Third Party Libs ####
import warnings

#### Comfy Libs ####
import nodes
import folder_paths

#### My Nodes #####
from EUP.nodes.sampler import Tiled_KSampler, Tiled_KSamplerAdvanced

#### Services ####
from EUP.services.lat_upsc_pixel import LatentUpscalerPixelSpaceService
from EUP.services.tiling import ULtimate_16KANDUPTilingService
from EUP.services.image import ImageService

'''
try:                    
    UpscaleCustomSample = nodes.NODE_CLASS_MAPPINGS['UltimateSDUpscaleCustomSample']
except Exception as e:
    print("Error importing upscaler:", e)
    UpscaleCustomSample = None
    if 'UltimateSDUpscaleCustomSample' not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("'UltimateSDUpscaleCustomSample' node (from comfyui_ultimatesdupscale) isn't installed.")
'''

class PixelTiledKSampleUpscalerService():
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars,
                 denoise, upscale_model_opt=None, hook_opt=None, tile_cnet_opt=None, tile_size=512, tile_cnet_strength=1.0, overlap=64
                 ):
        
        #### Services ####
        self.lupsService = LatentUpscalerPixelSpaceService()
        
        #### Parameters ####
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, denoise,
        self.scale_method = scale_method
        self.vae = vae
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiledKsample(self, latent, images):

        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, denoise = self.params

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return Tiled_KSampler().sample(
            model=model, 
            seed=seed,  
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
            sampler_advanced_pars=sampler_advanced_pars,
            denoise=denoise,
            )[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space2(samples, self.scale_method, upscale_factor, self.vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model2(samples, self.scale_method, self.upscale_model,
                                                          upscale_factor, self.vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscaleShape(self, step_info, samples, w, h, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_shape2(samples, self.scale_method, w, h, self.vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model_shape2(samples, self.scale_method,
                                                                self.upscale_model, w, h, self.vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent
    

class AdvancedPixelTiledKSampleUpscalerService():
    def __init__(self, scale_method, model, vae, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars,
                 start_at_step, end_at_step,return_with_leftover_noise, upscale_model_opt=None, hook_opt=None, tile_cnet_opt=None, tile_size=512, 
                 tile_cnet_strength=1.0, overlap=64
                 ):
        
        #### Services ####
        self.lupsService = LatentUpscalerPixelSpaceService()
        
        #### Parameters ####
        self.params = model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, start_at_step, end_at_step, return_with_leftover_noise
        self.scale_method = scale_method
        self.vae = vae
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiledKsample(self, latent, images):

        model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, sampler_advanced_pars, start_at_step, end_at_step, return_with_leftover_noise = self.params

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=images,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return Tiled_KSamplerAdvanced().sample(
            model=model, 
            add_noise=add_noise,
            seed=seed,  
            steps=steps, 
            cfg=cfg, 
            sampler_name=sampler_name,
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent_image=latent,
            sampler_advanced_pars=sampler_advanced_pars,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
            )[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space2(samples, self.scale_method, upscale_factor, self.vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model2(samples, self.scale_method, self.upscale_model,
                                                          upscale_factor, self.vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscaleShape(self, step_info, samples, w, h, save_temp_prefix=None):

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_shape2(samples, self.scale_method, w, h, self.vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                self.lupsService.latent_upscale_on_pixel_space_with_model_shape2(samples, self.scale_method,
                                                                self.upscale_model, w, h, self.vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        refined_latent = self.tiledKsample(upscaled_latent, upscaled_images)

        return refined_latent


'''
#from custom_nodes.comfyui-detail-daemon.nodes import DetailDaemon
from comfy_extras.nodes_custom_sampler import KSamplerSelect

class ULtimate_16KANDUPUpscalerService():

    UPSCALER_SETTINGS = {
        "1K": {
            "ultimateUpscaler" : {
                "cfg" : 3.5,
                "denoise" : 0.35,
                "tile_width" : 768,
                "tile_height" : 768,
                "force_uniform_tiles" : True,
                "tiled_decode" : True,
            },
            "deamon_Sampler" : {
                "detail_amount" : 0.50,
                "start" : 0.10,
                "end" : 0.90,
                "bias" : 0.50,
                "exponent" : 1.0,
                "start_offset" : 0.0,
                "end_offset" : 0.0,
                "fade" : 0.0,
                "smoooth" : True,
                "cfg_scale_override" : 3.5,
             },
        },
        "2K": {
            "ultimateUpscaler" : {
                "cfg" : 4.5,
                "denoise" : 0.35,
                "tile_width" : 768,
                "tile_height" : 768,
                "force_uniform_tiles" : True,
                "tiled_decode" : True,
            },
            "deamon_Sampler" : {
                "detail_amount" : 0.50,
                "start" : 0.10,
                "end" : 0.90,
                "bias" : 0.50,
                "exponent" : 1.0,
                "start_offset" : 0.0,
                "end_offset" : 0.0,
                "fade" : 0.0,
                "smoooth" : True,
                "cfg_scale_override" : 4.5,
             },
        },
        "4K": {
            "ultimateUpscaler" : {
                "cfg" : 3.5,
                "denoise" : 0.25,
                "tile_width" : 768,
                "tile_height" : 768,
                "force_uniform_tiles" : True,
                "tiled_decode" : True,
            },
            "deamon_Sampler" : {
                "detail_amount" : 0.45,
                "start" : 0.10,
                "end" : 0.90,
                "bias" : 0.50,
                "exponent" : 1.0,
                "start_offset" : 0.0,
                "end_offset" : 0.0,
                "fade" : 0.0,
                "smoooth" : True,
                "cfg_scale_override" : 3.5,
             },
        },
        "8K": {
            "ultimateUpscaler" : {
                "cfg" : 2.5,
                "denoise" : 0.15,
                "tile_width" : 1280,
                "tile_height" : 1280,
                "force_uniform_tiles" : True,
                "tiled_decode" : True,
            },
            "deamon_Sampler" : {
                "detail_amount" : 0.35,
                "start" : 0.10,
                "end" : 0.90,
                "bias" : 0.50,
                "exponent" : 1.0,
                "start_offset" : 0.0,
                "end_offset" : 0.0,
                "fade" : 0.0,
                "smoooth" : True,
                "cfg_scale_override" : 2.5,
             },
        },
        "16K": {
            "ultimateUpscaler" : {
                "cfg" : 1.5,
                "denoise" : 0.05,
                "tile_width" : 1280,
                "tile_height" : 1280,
                "force_uniform_tiles" : True,
                "tiled_decode" : True,
            },
            "deamon_Sampler" : {
                "detail_amount" : 0.25,
                "start" : 0.10,
                "end" : 0.90,
                "bias" : 0.50,
                "exponent" : 1.0,
                "start_offset" : 0.0,
                "end_offset" : 0.0,
                "fade" : 0.0,
                "smoooth" : True,
                "cfg_scale_override" : 1.5,
             },
        },
    }

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

        ## Services ## 
        self.tiligService = ULtimate_16KANDUPTilingService()
        self.imageService = ImageService()

    def getSaveImagePath(self, filename_prefix, output_dir):
        index = 1
        while True:
            filename_base = f"{filename_prefix}_{index:06d}"
            filename = f"{filename_base}.pkl"
            full_output_folder = output_dir
            full_path = os.path.join(full_output_folder, filename)
            if not os.path.exists(full_path):
                return full_output_folder, filename, filename_prefix
            index += 1
    
    def upscale(self, image, model, positive, negative, vae, upscale_by, seed, steps,
                sampler_name, scheduler, upscale_model, pickle_prefix, tile_num, mode_type,
                mask_blur, tile_padding, seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, 
                custom_sampler=None, custom_sigmas=None):
        
        width, height, count = self.imageService.getImageSize(image)

        averagSize = (width + height) / 2
        
        deamonSampler = nodes.NODE_CLASS_MAPPINGS['DetailDaemonSamplerNode']()
                       
        sdUpscaler = UpscaleCustomSample()

        for upscaleNum in range(upscale_by / 2):
            chosenImage = image
            if averagSize <= 1024:
                nomrmalSampler = KSamplerSelect().get_sampler(sampler_name)
                deamonSettings = self.UPSCALER_SETTINGS.get("1K").get("deamon_Sampler")
                customSampler = deamonSampler.go(
                    sampler=nomrmalSampler,
                    detail_amount=deamonSettings.get("detail_amount"),
                    start=deamonSettings.get("start"),
                    end=deamonSettings.get("end"),
                    bias=deamonSettings.get("bias"),
                    exponent=deamonSettings.get("exponent"),
                    start_offset=deamonSettings.get("start_offset"),
                    end_offset=deamonSettings.get("end_offset"),
                    fade=deamonSettings.get("fade"),
                    smooth=deamonSettings.get("smoooth"),
                    cfg_scale_override=deamonSettings.get("cfg_scale_override")
                )
                upscalerSettings = self.UPSCALER_SETTINGS.get("1K").get("ultimateUpscaler")
                chosenImage = sdUpscaler.upscale(
                    image=chosenImage, model=model, positive=positive, negative=negative, vae=vae,upscale_by=upscale_by, seed=seed, steps=steps,
                    cfg=upscalerSettings.get("cfg"), sampler_name=sampler_name, scheduler=scheduler, denoise=upscalerSettings.get("denoise"),
                    mode_type=mode_type, tile_width=upscalerSettings.get("tile_width"), tile_height=upscalerSettings.get("tile_height"),
                    mask_blur=mask_blur, tile_padding=tile_padding, seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise, 
                    seam_fix_mask_blur=seam_fix_mask_blur, seam_fix_width=seam_fix_width, seam_fix_padding=seam_fix_padding, 
                    force_uniform_tiles=upscalerSettings.get("force_uniform_tiles"), tiled_decode=upscalerSettings.get("tiled_decode"),
                    upscale_model=upscale_model, custom_sampler=customSampler, custom_sigmas=custom_sigmas
                    )

        full_output_folder, filename, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

'''


    