from ait.ait import AIT

if __name__ == "__main__":
    ait = AIT()
    ait.load("/home/user/ait_modules/unet_64_1024_1_1.so", "runwayml/stable-diffusion-v1-5", "unet")
    ait.test_unet()
    ait = AIT()
    ait.load("/home/user/ait_tmp/tmp/v1_vae_64_1024/test.so", "runwayml/stable-diffusion-v1-5", "vae")
    ait.test_vae()
    ait = AIT()
    ait.load("/home/user/ait_tmp/v1_clip_1/test.so", "runwayml/stable-diffusion-v1-5", "clip")
    ait.test_clip()
    ait = AIT()
    ait.load("/home/user/ait_tmp/v1_controlnet_512_512_1/test.so", "lllyasviel/sd-controlnet-canny", "controlnet")
    ait.test_controlnet()
    # ait = AIT()
    # ait.load_compvis("/home/user/ait_modules/unet_64_1024_1_1.so", "/home/user/checkpoints/v1-5-pruned-emaonly.ckpt", "unet")
    # ait.test_unet()