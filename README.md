# 3D Object Reconstruction from 2D images

This repository contains the source code for the paper [Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images](https://arxiv.org/abs/1901.11153). The follow-up work [Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images](https://arxiv.org/abs/2006.12250) has been published in *International Journal of Computer Vision (IJCV)*.

![Overview](https://infinitescript.com/wordpress/wp-content/uploads/2019/04/Pix2Vox-Overview.jpg)

> This project is an extension on the [original work](https://github.com/hzxie/Pix2Vox.git) by [Haozhe Xie](https://github.com/hzxie) exploring improvements to the algorithm to generate better optimized 3D images.


## Datasets

We use the [ShapeNet](https://www.shapenet.org/) and [Pix3D](http://pix3d.csail.mit.edu/) datasets in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
- Pix3D images & voxelized models: http://pix3d.csail.mit.edu/data/pix3d.zip

<!-- ## Pretrained Models

The pretrained models on ShapeNet are available as follows:

- [Pix2Vox-A](https://gateway.infinitescript.com/?fileName=Pix2Vox-A-ShapeNet.pth) (457.0 MB)
- [Pix2Vox-F](https://gateway.infinitescript.com/?fileName=Pix2Vox-F-ShapeNet.pth) (29.8 MB) -->

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/chiefoleka/3d-object-reconstruction-using-2d-images 3d-reconstruction
```

#### Install Python Denpendencies

```
cd 3d-reconstruction
pip install -r requirements.txt
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/path/to/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/path/to/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/path/to/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/path/to/Datasets/Pix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/path/to/Datasets/Pix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/path/to/Datasets/Pix3D/model/%s/%s/%s.binvox'
```

## Get Started

To train the model, you can simply use the following command:

```
python3 runner.py
```

To test model, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

## License

This project is open sourced under MIT license.
