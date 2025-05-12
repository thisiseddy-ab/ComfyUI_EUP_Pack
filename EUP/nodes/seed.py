import random

class GenerateSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate"
    CATEGORY = "Utility"

    def generate(self):
        return (random.randint(0, 4294967295),)
    
NODE_CLASS_MAPPINGS = {
    "EUP - Generate Seed": GenerateSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
