# The following two lines of code address this problem --- OMP: Error
#  OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import transforms as transforms
from dataloader import lunanod
import pandas as pd


# enter the 'resnet' class first
from models.aff_resnet_3d import *

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


#the test model
def load_data(test_data_path, preprocess_path, fold, batch_size, num_workers):

    crop_size = 32
    black_list = []


    pix_value, npix = 0, 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list:
                continue
            data = np.load(os.path.join(preprocess_path, file_name))
            pix_value += np.sum(data)
            npix += np.prod(data.shape)
    pix_mean = pix_value / float(npix)
    pix_value = 0
    for file_name in os.listdir(preprocess_path):
        if file_name.endswith('.npy'):
            if file_name[:-4] in black_list: continue
            data = np.load(os.path.join(preprocess_path, file_name)) - pix_mean
            pix_value += np.sum(data * data)
    pix_std = np.sqrt(pix_value / float(npix))
    print(f'pix_mean, pix_std: {pix_mean}, {pix_std}')
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((pix_mean), (pix_std)),
    ])

    # load data list
    test_file_name_list = []# this will be used later in the code
    test_label_list = []
    test_feat_list = []

    data_frame = pd.read_csv('./data/annotationdetclsconvfnl_v3.csv',
                             names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

    all_list = data_frame['seriesuid'].tolist()[1:]
    label_list = data_frame['malignant'].tolist()[1:]
    crdx_list = data_frame['coordX'].tolist()[1:]
    crdy_list = data_frame['coordY'].tolist()[1:]
    crdz_list = data_frame['coordZ'].tolist()[1:]
    dim_list = data_frame['diameter_mm'].tolist()[1:]
    # test id
    test_id_list = []
    for file_name in os.listdir(test_data_path + str(fold) + '/'):

        if file_name.endswith('.mhd'):
            test_id_list.append(file_name[:-4])
    mxx = mxy = mxz = mxd = 0
    for srsid, label, x, y, z, d in zip(all_list, label_list, crdx_list, crdy_list, crdz_list, dim_list):
        mxx = max(abs(float(x)), mxx)
        mxy = max(abs(float(y)), mxy)
        mxz = max(abs(float(z)), mxz)
        mxd = max(abs(float(d)), mxd)
        if srsid in black_list:
            continue
        # crop raw pixel as feature
        data = np.load(os.path.join(preprocess_path, srsid + '.npy'))
        bgx = int(data.shape[0] / 2 - crop_size / 2)
        bgy = int(data.shape[1] / 2 - crop_size / 2)
        bgz = int(data.shape[2] / 2 - crop_size / 2)
        data = np.array(data[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size])
        y, x, z = np.ogrid[-crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2, -crop_size / 2:crop_size / 2]
        mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
        feat = np.zeros((crop_size, crop_size, crop_size), dtype=float)
        feat[mask] = 1
        if srsid.split('-')[0] in test_id_list:
            test_file_name_list.append(srsid + '.npy')
            test_label_list.append(int(label))
            test_feat_list.append(feat)
    for idx in range(len(test_feat_list)):
        test_feat_list[idx][-1] /= mxd

    test_set = lunanod(preprocess_path, test_file_name_list, test_label_list, test_feat_list, train=False,
                       download=True,
                       transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

# add 'grad_cam'  class
def cam_visualization_single_cube(model, test_loader):# defining the test model
    model.eval()


    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda() # input[1,1,32,32,32] [batch, channel, Z,Y,X]

        cam_extractor = SmoothGradCAMpp(model, target_layer='layer4',
                                        input_shape=(32, 32, 32))  # 'input_shape' is used to calculate batch_size?

        out = model(inputs)
        class_idx = out.squeeze(0).squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, out)
        # activation_map = cam_extractor(class_idx = 0, scores = out) # class_idx=0 or 1
        activation_map_numpy = activation_map[0].cpu().numpy()
        print(f'activation_map_numpy:{activation_map_numpy}')
        print(f'activation_map_numpy.shape:{activation_map_numpy.shape}')
        result = overlay_mask(to_pil_image(inputs), to_pil_image(activation_map[0], mode='F'), alpha=0.5)

if __name__ == '__main__':
    fold = 5
    batch_size = 1
    num_workers = 0
    test_data_path = './data/LUNA/subset'
    preprocess_path = './data/LUNA/crop_v3'
    # model_path = './resnet50_checkpoint/ckpt.t7'
    # net = load_module(model_path)
    net = resnet18(fuse_type='DAF', small_input=False).cuda().train()  # liu 11.20
    test_data_loader = load_data(test_data_path, preprocess_path, fold, batch_size, num_workers)
    cam_visualization_single_cube(net, test_data_loader)
