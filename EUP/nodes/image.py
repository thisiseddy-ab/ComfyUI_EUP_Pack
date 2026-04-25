#### My Services's #####
from EUP.services.image import ImageService
from EUP.nodes.aspect_ratio import AspectRatioService, CustomAspectRatio

class GetImageSize:

    def __init__(self):
        self.imageService = ImageService()
        

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "EUP - Ultimate Pack/image"

    def execute(self, image):
        return self.imageService.getImageSize(image)
    

class CustomImageSize:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "biggest_size": ("INT", {"default": 1352, "min": 125, "max": 8192}),
                "width": ("INT", {"default": 1352, "min": 125, "max": 8192}),
                "height": ("INT", {"default": 1352, "min": 125, "max": 8192}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio")
    FUNCTION = "execute"
    CATEGORY = "EUP - Ultimate Pack/image"

    def execute(self, biggest_size, width, height):
        closest_aspect_ratio = AspectRatioService.find_closest(
            width=width,
            height=height,
            aspect_ratios=CustomAspectRatio.ASPECT_RATIO,
        )
        new_width, new_height, aspect_ratio_str = AspectRatioService.calculate_dimensions(
            biggest_size=biggest_size,
            aspect_ratio=closest_aspect_ratio,
        )

        return (new_width, new_height, aspect_ratio_str)
    

NODE_CLASS_MAPPINGS = {
    "EUP - Get Image Size": GetImageSize,
    "EUP - Custom Image Size": CustomImageSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
