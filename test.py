from ait import AIT

if __name__ == "__main__":
    ait = AIT()
    ait.load("/home/user/ait_modules/unet_64_1024_1_1.so", "runwayml/stable-diffusion-v1-5", "unet")
    ait.test_unet()
