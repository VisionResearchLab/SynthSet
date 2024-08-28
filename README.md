Source code for the method described in the paper "SynthSet: Generative Diffusion Model for Semantic Segmentation in Precision Agriculture".

This repository is a fork from the extendable ClasSeg (https://github.com/aheschl1/ClasSegPipeline) package, and implements functionality using the "super_resolution" and "unstable_diffusion" extensions.

Please read the ClasSeg README for in depth details on how to use the pipeline.

**Segmentation Structure**
```
RAW_ROOT
|
---Dataset_<desc>_<id>
.              |
.              ---- imagesTr
.              .        | case_00000.png
.              .        | case_000101.png
.              ---- labelsTr
.              .        | case_00000.png
.              .        | case_00101.png
---Dataset_<desc2>_<id2>
|
```


# Running Training
To run training, you need to run the following command:
```classegTrain -d <dataset id> -f <fold to train> -ext <extension name> -m <model name> -n <experiment name>```
To get more info, run ```classegTrain --help```

# Running Inference
This portion is going to face a lot of changes in the future. For now, you can run inference with the following command:
```classegInfer -d <dataset id> -f <fold to infer> -ext <extension name> -i <input_folder>``` What happens if you are SSL and have no input? Too bad. Put something random for -i.
Run ```classegInfer --help``` for more details, and up to date information.
