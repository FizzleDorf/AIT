from .AITemplate.AITemplate import AITemplateLoader, AITemplateControlNetLoader, AITemplateVAEDecode, AITemplateVAEEncode, VAEEncodeForInpaint

NODE_CLASS_MAPPINGS = {
    "AITemplateLoader": AITemplateLoader,
    "AITemplateControlNetLoader": AITemplateControlNetLoader,
    "AITemplateVAEDecode": AITemplateVAEDecode,
    "AITemplateVAEEncode": AITemplateVAEEncode,
    "AITemplateVAEEncodeForInpaint": VAEEncodeForInpaint,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AITemplateLoader": "Load AITemplate",
    "AITemplateControlNetLoader": "Load AITemplate (ControlNet)",
    "AITemplateVAELoader": "Load AITemplate (VAE)",
    "AITemplateVAEDecode": "VAE Decode (AITemplate)",
    "AITemplateVAEEncode": "VAE Encode (AITemplate)",
    "AITemplateVAEEncodeForInpaint": "VAE Encode (AITemplate, Inpaint)",
}
