## STAD (Student-Teacher Anomaly Detection)
Pytorch implementation of the paper [Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings](https://arxiv.org/abs/1911.02357) by Paul Bergmann

## Requirement
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
- [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))
- docker image: [taikiinoue45/pytorch:tensorrt](https://hub.docker.com/layers/taikiinoue45/pytorch/tensorrt/images/sha256-917e3547bc77b0ed5d49225e5166a08d98504d91cf11d209a4de41ab4fbb8ab9?context=explore)

## Roadmap
#### Accelerate Training and Inference
- [x] TensorRT support
- [ ] [Fast dense local feature extraction](https://www.dfki.de/fileadmin/user_upload/import/9245_FastCNNFeature_BMVC.pdf)
#### Improve the Performance
- [ ] An ensemble of student networks
- [ ] Multiscale prediction
