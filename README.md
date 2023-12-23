# CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment

This is the official code of the CVPR 2023 paper (`Highlight presentation, acceptance rate: 2.5% of submitted papers`) *CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment* [[CVPR Version]](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.html) [[arXiv Version]](https://arxiv.org/abs/2303.05725).


## !!! See Also
-  [**Awesome AI Sign Language Papers**](https://github.com/binbinjiang/SL_Papers). 
If you are new or interested in AI sign language field, we **highly recommend** you browse this [repository](https://github.com/binbinjiang/SL_Papers). We have collected papers on AI Sign Language (SL) comprehensively. For easy searching and viewing, we have categorized them according to different criteria (by time, type of research, institution, etc.). Feel free to add contents and submit updates.
- **Extensiton Work**: The proposed novel cross-modal transformation in this work has been successfully applied to a protein design (an impotant cross-modal protein task in AI life) framework, which achieves an excellent performance. (e.g., [MMDesign: Multi-Modality Transfer Learning for Generative Protein Design](https://arxiv.org/pdf/2312.06297.pdf)) 
<!-- Please refer to this [repo]() and [paper]() -->
- Stay tuned for more of our work related to this work!

## News
- 2023.03.21 -> This work was selected as a **highlight** by CVPR 2023 (Top **2.5%** of submissions, **10%** of accepted papers)

- 2023.02.28 -> This work was accepted to CVPR 2023 (9155 submissions, and accepted 2360 papers, 25.78% accptance rate)


##  Proposed CVT-SLR Framework

<img src=".\imgs\framework.jpg" alt="framework" style="zoom: 80%;" />

For more details, please refer to our [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.pdf).

## Prerequisites
### Dependencies
As a **prerequisite**, you are suggested to create a brand new *conda environment* firstly. A reference python dependency packages could be installed as follows :

 (1) python==3.8.16
 
 (2) torch==1.12.0+cu116, pls see [Pytorch official website](https://pytorch.org/get-started/locally/) 
 
 (3) PyYAML==6.0 
 
 (4) tqdm==4.64.0

 (5) opencv-python==4.2.0.32

 (6) scipy==1.4.1

**F.Y.I**: Not all are required and appropriate, it depends on your actual situation.


Besides, you must install ctcdecode==0.4 for beam search decode, pls see this [repo](https://github.com/parlance/ctcdecode) in detail. Run the following command to install ctcdecode:

```bash
cd ctcdecode && pip install .
```


### Datasets
For **data preparation**, please download [phoenix2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) dataset and [phoenix2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) dataset in advance. *After extracting, it is suggested to make a soft link toward downloaded dataset*.


For more details on data preparation and prerequisites, please refer to this [repo](https://github.com/ycmin95/VAC_CSLR). We are grateful for the foundation that their work has given us. 

**NB**: 1) Please refer to the above-mentioned repo for dataset extracting to the [./dataset](./dataset) directory. 2) Resize the original sign images from 210x260 to 256x256 for augmentation, and the generated gloss dict and resized image sequences are saved in [./preprocess](./preprocess) for your reference. 3) We didn't use [sclite library](https://github.com/kaldi-asr/kaldi) for evaluation (this library maybe tricky  to install) but use pure python implemented evaluation tools instead, see [./evaluation](./evaluation).


## Configuration Setting
According to your actual situations, update the configurations in [./configs/phoenix14.yaml](./configs/phoenix14.yaml) and [./configs/cvtslt_eval_config.yaml](./configs/cvtslt_eval_config.yaml). Especially, focus on the hyper-parameters such as *dataset_root*, *evaluation_dir*, *work_dir*.


## Demo Evaluation

​We provide the pretrained CVT-SLR models for inference, as:

Firstly, download checkpoints to [./trained_models](./trained_models) directory from the corresponding links as follows. Then, evaluate the pretrained model using script as:

**->** [Option 1] Using AE based configuration:

`python run_demo.py --work-dir ./out_cvpr/cvtslt_2/ --config ./configs/cvtslt_eval_config.yaml --device 1 --load-weights ./trained_models/cvtslt_model_dev_19.87.pt --use_seqAE AE`

Evaluation results: test 20.17%, dev 19.87%


**->** [Option 2] Using VAE based configuration:

`python run_demo.py --work-dir ./out_cvpr/cvtslt_1/ --config ./configs/cvtslt_eval_config.yaml --device 1 --load-weights ./trained_models/cvtslt_model_dev_19.80.pt --use_seqAE VAE`

Evaluation results: test 20.06%, dev 19.80%


The updated evaluation results (WER %) and download links:

| Group                | Models                | Dev | Test |                       Trained Checkpoints                       |
|:---------------------- | :---------------------- | :--------: | :---------: | :----------------------------------------------------------: |
|Group 1 (single-cue)  | SubUNet     |    40.8    |    40.7     | -|
|  | Staged-Opt     |    39.4    |    38.7     | -|
|  | Align-iOpt     |    37.1    |    36.7     | -|
|  | DPD+TEM     |    35.6    |    34.5     | -|
|  | Re-Sign     |    27.1    |    26.8     | -|
|  | SFL     |    26.2    |    26.8     | -|
|  | DNF     |    23.8    |    24.4     | -|
|  | FCN    |    23.7    |   23.9     | -|
|  | VAC    |    21.2    |   22.3     | -|
|  | CMA    |    21.3    |   21.9     | -|
|  | SFL    |    24.9    |   25.3     | -|
|  | VL-SLT    |    21.9    |   22.5     | -|
|  | SMKD    |    20.8    |   21.0     | -|
|Group 2 (multi-cue)  | DNF     |    23.1    |    22.9     | -|
|  | STMC     |    21.1    |    20.7     | -|
|  | C2SLR     |    20.5    |    20.4     | -|
|Group 3 (Ours) | CVT-SLR w/ AE     |    19.87    |    20.17     | [[Baidu]](https://pan.baidu.com/s/1AE8L9M3u080L_T5G6Aqsvg?pwd=k42q) (pwd/提取码: k42q) or [[GoogleDrive]](https://drive.google.com/file/d/1ErXIfxCgSbrKeGln3_xezBWT4C-3jS8A/view?usp=sharing) |
| | CVT-SLR w/ VAE     |    19.80    |    20.06     | [[Baidu]](https://pan.baidu.com/s/1vF2G07wjX6f-gpxVZgPEOg?pwd=0kga) (pwd/提取码: 0kga) or [[GoogleDrive]](https://drive.google.com/file/d/1_2-zBi3dop8JIJYrl_sufUWBDR7DAawl/view?usp=sharing)|


**NB:** please refer to our [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.pdf) for more details.

## Visualization
- Saliency Maps
<img src=".\imgs\saliency_map_case.png" alt="framework" style="zoom: 80%;" />
We visualize the key parts of the sign video frames in focus by using Grad-CAM. To implement this function, you can use the open python tool as:
```
import pytorch_grad_cam
```

- Cross-modal Alignment Matrices
<img src=".\imgs\alignment_matrices_case.png" alt="framework" style="zoom: 30%;" />
To generate the cross alignment matrices, here are some hints as:

```
a = ret["conv_logits"].squeeze(1)
b = ret["sequence_logits"].squeeze(1)
T = 1
simi_matric = softmax(T*(a @ b.T))
```

## Citation

If you find this repository useful, please consider citing:

```
@inproceedings{zheng2023cvt,
  title={Cvt-slr: Contrastive visual-textual transformation for sign language recognition with variational alignment},
  author={Zheng, Jiangbin and Wang, Yile and Tan, Cheng and Li, Siyuan and Wang, Ge and Xia, Jun and Chen, Yidong and Li, Stan Z},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23141--23150},
  year={2023}
}
```

<!--
@article{zheng2023cvt,
  title={CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment},
  author={Zheng, Jiangbin and Wang, Yile and Tan, Cheng and Li, Siyuan and Wang, Ge and Xia, Jun and Chen, Yidong and Li, Stan Z},
  journal={arXiv preprint arXiv:2303.05725},
  year={2023}
}
-->