import uuid

class GenerateUUID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}  # No inputs
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("uuid",)
    FUNCTION = "generate"
    CATEGORY = "Utility"

    def generate(self):
        return (str(uuid.uuid4()),)
    


NODE_CLASS_MAPPINGS = {
    "EUP - Generate UUID": GenerateUUID,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}
