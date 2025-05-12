import os
import re

class CollectionImageSavePrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "collection": ("STRING", {"default": ""}),
                "prefix": (["txt2img", "img2img", "upscale"],),
                "prefix_extra": ("STRING", {"default": ""}),
                "uuid": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16-9"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "generate"
    CATEGORY = "Utility"

    def generate(self, collection, prefix, prefix_extra, uuid, aspect_ratio):
        collection = self.sanitizePathComponent(collection)
        prefix = self.sanitizePathComponent(prefix)
        prefix_extra = self.sanitizePathComponent(prefix_extra)
        uuid = self.sanitizePathComponent(uuid)
        aspect_ratio = self.sanitizePathComponent(aspect_ratio)

        safe_path = os.path.join(f"{uuid}-{collection}", prefix, aspect_ratio, f"{prefix}-{prefix_extra}")
        return (safe_path,)
    
    def sanitizePathComponent(self, component: str) -> str:
        # Remove or replace invalid characters for Windows and general safety
        component = str(component).strip()
        component = component.replace(":", "-") 
        component = re.sub(r'[<>:"/\\|?*]', '-', component) 
        component = re.sub(r'\s+', '_', component) 
        return component[:100] 



NODE_CLASS_MAPPINGS = {
    "EUP - Collection Image Save Prefix": CollectionImageSavePrefix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
