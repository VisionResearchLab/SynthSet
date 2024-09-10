Source code for the method described in the paper "SynthSet: Generative Diffusion Model for Semantic Segmentation in Precision Agriculture".

This repository is a fork from the extendable ClasSeg (https://github.com/aheschl1/ClasSegPipeline) package, and implements functionality using the "super_resolution" and "unstable_diffusion" extensions. Please read the ClasSeg README for in depth details on how to use the pipeline.

This pipeline includes image-mask pair generation, as well as super resolution for doubling the resolution from 128^2 to 256^2.
![good_images](https://github.com/user-attachments/assets/fa95bceb-ef7e-46c2-8743-2cf24b42c865)

Super resolution is highly effective for maintining efficacy between images and masks.
![super_samples](https://github.com/user-attachments/assets/96b2f717-f56e-43df-a28b-39b18dc560b1)

This code defines the following pipeline, where the third row is acheived through setting mode to "concat" in the config. The bottom row is acheived through setting "gan_weight" > 0  in the config:
![diffusion_pipeline](https://github.com/user-attachments/assets/9f7e2851-2a82-4c09-a269-7612011a3ec0)
