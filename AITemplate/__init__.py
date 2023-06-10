from .AITemplate import AITemplateLoader, AITemplateControlNetLoader, AITemplateVAEDecode

NODE_CLASS_MAPPINGS = {
    "AITemplateLoader": AITemplateLoader,
    "AITemplateControlNetLoader": AITemplateControlNetLoader,
    "AITemplateVAEDecode": AITemplateVAEDecode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AITemplateLoader": "Load AITemplate",
    "AITemplateControlNetLoader": "Load AITemplate (ControlNet)",
    "AITemplateVAEDecode": "VAE Decode (AITemplate)",
}
