# Document_Image_Dewarping

The code for "[Foreground and Text-lines Aware Document Image Rectification](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Foreground_and_Text-lines_Aware_Document_Image_Rectification_ICCV_2023_paper.pdf)", ICCV, 2023.

## Training Dataset
We use the Doc3D dataset for training. You can download the dataset on 
[DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) or [doc3D-dataset](https://github.com/fh2019ustc/doc3D-dataset).

## Evaluation Dataset
We evaluate on two datasets [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) and [DIR300](https://github.com/fh2019ustc/DocGeoNet).

## Environment & Installation

The code was developed against the environment below (Linux, Python 3.10):

```text
torch==1.13.0+cu117   # PyTorch 1.13.0, CUDA 11.7 build
opencv-python==4.8.1.78
numpy==1.26.0
mmcv-full==1.7.1      # OpenMMLab mmcv 1.x (provides mmcv.runner.BaseModule, mmcv.cnn.ConvModule)
Pillow==9.4.0
```

`networks/cross_attn.py` imports `from mmcv.runner import BaseModule` and
`from mmcv.cnn import ConvModule`. `mmcv.runner` only exists in the **mmcv 1.x
("mmcv-full")** series; it was removed in mmcv 2.x. Pick one of the two options
below depending on your machine.

### Option A — Reproduce the paper's environment (recommended)

Create an isolated environment and install the pinned versions. `cp310` wheels
exist for every dependency, so this works on Python 3.10. The `torch==1.13.0+cu117`
wheel bundles its own CUDA 11.7 runtime, so it runs on any host whose NVIDIA
driver is recent enough (driver `>= 515`, which is satisfied by any CUDA 12.x /
NGC 23.09 host, driver `535+`) — the host CUDA toolkit version does not need to match.

```bash
# inside the NGC 23.09 container (or any CUDA 12.x host)
deactivate 2>/dev/null          # leave any active env
unset PYTHONPATH                # avoid inheriting the container's site-packages
python3.10 -m venv dewarp       # PLAIN venv (do NOT use --system-site-packages for Option A)
source dewarp/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Important — use an *isolated* venv for Option A.** This installs
> `torch 1.13.0+cu117`, which must **replace** the container's torch 2.1. If you
> install into the container directly (or into a `--system-site-packages` venv),
> pip swaps torch's Python files but the container's torch 2.x compiled extension
> (`torch._C`) can still get loaded, giving:
> `TypeError: cannot set __module__ attribute of immutable type 'torch._C.DisableTorchFunctionSubclass'`
> (`DisableTorchFunctionSubclass` is a torch 2.x symbol that does not exist in
> torch 1.13 — its presence proves the env is mixed). A plain venv has no torch to
> collide with, so it avoids this.
>
> **Being inside an activated venv is not enough.** NGC containers usually set the
> `PYTHONPATH` environment variable to their system `dist-packages`, and those
> entries are added to `sys.path` *ahead of* your venv even when it is activated —
> so `import torch` can still resolve to the container's torch 2.1. Always
> `unset PYTHONPATH` (as above) and verify with the `torch.__file__` check below
> that torch resolves to a path *inside your venv*.

Verify you are on the clean torch before running anything else:

```bash
python -c "import torch; print(torch.__version__, torch.__file__)"
# expected: 1.13.0+cu117  /.../dewarp/lib/python3.10/site-packages/torch/__init__.py
```

The `torch==1.13.0+cu117` wheel and `mmcv-full 1.7.1` both come from the index
URLs declared in `requirements.txt`; no source compilation or GPU-toolkit match
is required.

### Option B — Use the container's native PyTorch 2.1 / CUDA 12.2

If you prefer to keep the NGC 23.09 container's bundled `torch 2.1.0a0` /
CUDA 12.2, do **not** reinstall torch. `mmcv-full 1.7.1` has no wheel for
torch 2.x, but the patch release **`mmcv-full 1.7.2`** does (cu121/torch2.1.0,
`cp310`), and it keeps the same `mmcv.runner` / `mmcv.cnn` API, so **no code
changes are needed**:

```bash
# keep the container's torch 2.1 — do NOT reinstall torch
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install opencv-python==4.8.1.78 numpy==1.26.0 Pillow==9.4.0
```

Note: this prebuilt wheel targets the stock `torch 2.1.0` / CUDA 12.1 build,
whereas NGC 23.09 ships a custom `torch 2.1.0a0` / CUDA 12.2 build, so its
compiled CUDA ops can occasionally hit an ABI mismatch. If you see errors when
importing `mmcv` ops, fall back to Option A, which avoids that risk entirely.

## Inference
Please download the pre-trained model from 
[Google Drive](https://drive.google.com/drive/folders/1UWL7wWSCcyhHuWLSKQRI9g2_cp0M0aD-?usp=sharing) 
or [Baidu Cloud](https://pan.baidu.com/s/1JhEznQEjaVplPQww0CNbHA?pwd=p5yp). Then execute:
 
 `python predict.py --model_path /MODEL/PATH --img_path /BENCHMARK/DIR --save_path /SAVE/PATH`
 
## Evaluation

We follow the evaluation environment and code in [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 
and [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet).

For CER and ED metrics evaluation:

```text
Tesseract==5.0.1.20220118 (Windows)
pytesseract==0.3.8
```

The dewarped images can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PHyeZZF88-KzkeuV8YiKHJ_z9lC-3C9o) 
or [Baidu Cloud](https://pan.baidu.com/s/1Lq9tRbOM4nV-pQ9sbVfbww?pwd=y41i).
## Acknowledgement
Our methods and codes are inspired by many existing works, to which we would like to express special thanks to:

[DocUNet: Document Image Unwarping via A Stacked U-Net](https://www3.cs.stonybrook.edu/~cvl/content/papers/2018/Ma_CVPR18.pdf)

[DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D
Regression Networks](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/storage/paper.pdf)

[DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/pdf/2110.12942.pdf)

[Revisiting Document Image Dewarping by Grid Regularization](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)

[Geometric Representation Learning for Document Image Rectification](https://arxiv.org/pdf/2210.08161.pdf)


## Citation
If our methods and code are helpful to you, please refer to the following BibTeX format for citation:
```
@inproceedings{li2023foreground,
  title={Foreground and Text-lines Aware Document Image Rectification},
  author={Li, Heng and Wu, Xiangping and Chen, Qingcai and Xiang, Qianjin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19574--19583},
  year={2023}
}
```


