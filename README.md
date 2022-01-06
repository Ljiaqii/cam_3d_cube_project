Note: I am currently trying to use CAM (Class Activation Mapping) for 3D CNN visualization, but there are still problems that have not been resolved. 
I would be very grateful if you can provide help and suggestions to solve activation_map_numpy is [[[nan]]] problem and activation_map_numpy.shape is (1, 1, 1) problem.
Besides, I also do not know why these happens. Thanks for your help!

#How to run:

To run this program, just run cam_3dcube.py
Other folders, such as ‘data’ folder, ‘models’ folder, or ‘dataloader.py’, ’transforms.py’, etc. , provide support for running ‘cam_3dcube.py’.
You only need to make the location of each folder in the project unchanged, the program can run!
PS：In this project, the ‘torchcam-author’ folder is not used, we use the command ‘pip install torchcam’ in anaconda to import torchcam package.

#The output information of the ‘cam_3dcube.py’ file is as follows:

E:\anaconda3.8\envs\pytorch1.8_cuda11.1_cudnn8.0.5_py37_ljq_naslung\python.exe D:/LiuJiaqi/cam_3d_cube_project/cam_3dcube.py
pix_mean, pix_std: 199.32785034179688, 15.390596152988984
self.test_data.shape, len(self.test_labels):
(1, 32, 32, 32) 1
Traceback (most recent call last):
activation_map_numpy:[[[nan]]]
activation_map_numpy.shape:(1, 1, 1)

#Error track:

I would be very grateful if you can provide help and suggestions to solve activation_map_numpy is [[[nan]]] problem and activation_map_numpy.shape is (1, 1, 1) problem. 
Besides, I also do not know why these happens. Thanks for your help!

Traceback (most recent call last):
activation_map_numpy:[[[nan]]]
activation_map_numpy.shape:(1, 1, 1)
  File "D:/LiuJiaqi/cam_3d_cube_project/cam_3dcube.py", line 141, in <module>
    cam_visualization_single_cube(net, test_data_loader)
  File "D:/LiuJiaqi/cam_3d_cube_project/cam_3dcube.py", line 129, in cam_visualization_single_cube
    result = overlay_mask(to_pil_image(inputs), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
  File "E:\anaconda3.8\envs\pytorch1.8_cuda11.1_cudnn8.0.5_py37_ljq_naslung\lib\site-packages\torchvision\transforms\functional.py", line 223, in to_pil_image
    raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))
ValueError: pic should be 2/3 dimensional. Got 5 dimensions.

Process finished with exit code 1

#My environment:

TorchCAM version: 0.3.1
PyTorch version: 1.8.0
OS: Microsoft Windows 10
Python version: 3.7.10
Is CUDA available: Yes
CUDA runtime version: 11.1.74
GPU models and configuration: GPU 0: GeForce RTX 3090
Nvidia driver version: 457.85
cuDNN version: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\cudnn_ops_train64_8.dll

#The version of each package:

torch:1.8.0
numpy:1.20.1+mkl
pandas:1.2.4
torchcam:0.3.1
torchvision:0.9.0
Pillow:8.4.0
pip:21.0.1
