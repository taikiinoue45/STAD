## STAD (Student-Teacher Anomaly Detection)
Pytorch implementation of the paper [Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings](https://arxiv.org/abs/1911.02357) by Paul Bergmann.   
  
Unsupervised anomaly segmentation is a desirable and challenging task in many domains of computer vision. Existing works mainly focus on generative algorithms such as Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs). These detect anomalies using per-pixel reconstruction errors or by evaluating the density obtained from the model’s probability distribution. This has been shown to be problematic due to inaccurate reconstructions or poorly calibrated likelihoods

In this paper, a novel framework is proposed for unsupervised anomaly segmentation based on student–teacher learning, and demonstrated the state-of-the-art performance on MNIST, CIFAR10 and MVTec AD dataset.

<br>

## 1. Re-Implementation
Unfortunately, the official implementation is not published. Therefore I tried to re-implement it from scratch.

### 1-1. MNIST Dataset
■ **Training the Student Network**  
My teacher network is the feature extraction module of torchvision.models.vgg19(pretrained=True). The architecture of a student network is exactly same to a teacher network, but pretrained=False. The training dataset consists of 6,742 images with 1 label, which are treated as the anomaly-free data, and all images with 0~9 labels are used as the test dataset. Since the image size is very small, the entire images are input to the student and teacher network. The teacher descriptors serve as surrogate labels, and the goal of training is to make the student network regress the surrogate labels. As you can see, the parameters in the teacher network are fixed, and the parameters in the student network are updated with the backpropagation. The implementation is available [here](https://github.com/TaikiInoue/STAD/blob/master/examples/MNIST.ipynb)

<p align="center">
<img src="https://user-images.githubusercontent.com/29189728/86548632-16ef8c80-bf78-11ea-99eb-85c7a5cea8b0.png" width=800>
</p>

<br>

■ **Anomaly Detection on MNIST Dataset**  
During inference, `0~9` images are input to the teacher and student, and the anomaly score is computed. I expect that `1` images result in the low anomaly scores, but other numbers are unseen during training, so the anomaly scores will be higher. As a result, it worked as I expected. It is relatively difficult to distinguish between `1` and `7`, which are similar in shape.

<p align="center">
<img src="https://user-images.githubusercontent.com/29189728/86548967-320ecc00-bf79-11ea-9697-33ab799aa7b7.png" width="600">
</p>

<br>

## Reproduce MVTec result
Next, I evaluated on the much more challenging MVTec AD dataset, which is quite similar to Somic dataset. The architecture of teacher and student is exactly same to the above MNIST experiment. Training dataset consists of 209 images with good label, and all images with good, broken_small, broken_large or contamination labels are used as the test dataset. The size of the original images is 900×900 , and 128×128 randomly cropped images are input to the networks, and the parameters in the student network are updated to regress the teacher descriptors.

<br>

## Roadmap
#### Accelerate Training and Inference
- [x] TensorRT support
- [ ] [Fast dense local feature extraction](https://www.dfki.de/fileadmin/user_upload/import/9245_FastCNNFeature_BMVC.pdf)
#### Improve the Performance
- [ ] An ensemble of student networks
- [ ] Multiscale prediction

<br>

## Requirement
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
- [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))
- docker image: [taikiinoue45/pytorch:tensorrt](https://hub.docker.com/layers/taikiinoue45/pytorch/tensorrt/images/sha256-917e3547bc77b0ed5d49225e5166a08d98504d91cf11d209a4de41ab4fbb8ab9?context=explore)
