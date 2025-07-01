import torch, random, json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2, math



def set_dataset_FFSC_SLH(txt_path, preprocess, args, logger, phase='train', aug=False):
    aug_probs = args.aug_probs # 0.8 #1.0 #

    if args.interpret:
        dataset = FFSC_dataset_SLH(txt_path, preprocess, mode_label=args.mode_label, aug_probs=0.,
                                   is_SLH=args.is_SLH,
                                   is_nWay=args.is_nWay,
                                   address_fn=True)
    else:
        if not aug:
            dataset = FFSC_dataset_SLH(txt_path, preprocess, mode_label=args.mode_label, aug_probs=0.,
                                       is_SLH=args.is_SLH,
                                       is_nWay=args.is_nWay)
        else:
            logger.info(f'augment dataset with probs of {aug_probs}')
            dataset = FFSC_dataset_SLH(txt_path, preprocess, mode_label=args.mode_label, aug_probs=aug_probs,
                                       is_SLH=args.is_SLH,
                                       is_nWay=args.is_nWay)
    print(f"successfully build {phase} dataset")

    if phase == 'train':
        data_loader_train = torch.utils.data.DataLoader(
            dataset, shuffle=True,
            batch_size=args.BATCH_SIZE,
            num_workers=args.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        return dataset, data_loader_train

    elif phase == 'val' or phase == 'test':
        data_loader_val = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=False,
            num_workers=args.NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )
        return dataset, data_loader_val

class FFSC_dataset_SLH(Dataset):
    def __init__(self, txt_path, preprocess, mode_label='global',
                 aug_probs=0.3, is_SLH=True, is_nWay=False,
                 address_fn=False):
        super().__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue
            imgs.append((words[0], words[1]))  # path, label
            # if dataset in ['ffsc', 'so-ffpp']:
            #     imgs.append((words[0], words[1])) # path, label
            # else:
            #     imgs.append((words[0], words[2])) # path, train_mode (e.g., train, val, test), label

        self.is_SLH = is_SLH
        self.is_nWay = is_nWay

        self.imgs = imgs
        self.label_mode = mode_label
        self.preprocess = preprocess
        self.aug_probs = aug_probs

        self.address = address_fn

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if random.random() < self.aug_probs:
            # img = self.perturbation(img)
            img = self.perturbation_jpeg(img)

        if self.preprocess is not None:
            img = self.preprocess(img)

        label_name = label
        label, mask = self.get_label_mask(label, self.label_mode)
        label_infer = self.get_label(label, self.label_mode)

        if self.is_SLH and not self.is_nWay:
            if not self.address:
                return img, label, mask, label_infer
            else:
                return img, label, mask, label_infer, fn
        elif not self.is_SLH and self.is_nWay:
            label_nWay = self.get_label_nWay(fn, label_name)
            return img, label_nWay
        # else:
        #     return img, label, mask, label_infer, label_name

    def __len__(self):
        return len(self.imgs)

    def get_label_nWay(self, img_path, label_name):
        DICT = ['diffae-age', 'StyleRes-age', 'diffae-gender', 'StyleGAN2_dis-gender','simswap', 'fsgan','BlendFace',
                'StyleRes-expr', 'fomm-expr', 'TPS-pose', 'fomm-pose']
        DICT_nWay = {
            'diffae-age': 1, 'StyleRes-age': 2,
            'diffae-gender': 3, 'StyleGAN2_dis-gender': 4,
            'simswap': 5, 'fsgan': 6, 'BlendFace': 7,
            'StyleRes-expr': 8, 'fomm-expr': 9,
            'TPS-pose': 10, 'fomm-pose': 11
        }
        if label_name == '0000000000':
            label_nWay = 0
        else:
            methods_name = img_path.split('/')[-3]
            if methods_name not in DICT:
                methods_name = img_path.split('/')[-2]
            label_nWay = DICT_nWay[methods_name]

        return label_nWay

    def get_label_mask(self, label, mode='global'):
        label = [int(char) for char in label]
        if mode == 'global': # for ffsc
            label_out = np.array(label[0:6])
            mask = np.ones_like(label_out)
            mask[label_out==2] = 0
        else:
            label_out = np.array(label)
            mask = np.ones_like(label_out)
            mask[label_out == 2] = 0
        return label_out, mask

    def get_label(self, label, mode='global'):
        label = [int(char) for char in label]
        label = [0 if num == 2 else -1 if num == 0 else num for num in label]
        if mode == 'global':
            label_out = np.array(label[0:6])
        else:
            label_out = np.array(label)
        return label_out

    def perturbation_jpeg(self, im):
        distortion = ['BW', 'GNC', 'GB', 'JPEG', 'RealJPEG']
        type = random.choice(distortion)
        # if random.random() < 0.6:
        #     level = random.randint(1,2)
        # else:
        #     level = 3
        if random.random() < 0.5:
            level = 1
        else:
            level = 2

        if type != 'RealJPEG':
            im = np.asarray(im)
            im = np.copy(im)
            im = np.flip(im, 2)

            dist_function, dist_param = self.get_perturb(type, level)
            im = dist_function(im, dist_param)
            im = Image.fromarray(np.flip(im, 2))

            return im
        else: # REAL JPEG
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)  # PIL转cv2
            # pdb.set_trace()
            # # quality = random.randint(75, 100)
            # quality = random.choice([75, 80, 90])
            quality = random.choice([70, 80, 90])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', im, encode_param)[1]
            im = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
            im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            return im

    def get_perturb(self, type, level):
        def get_distortion_function(type):
            func_dict = dict()  # a dict of function
            func_dict['BW'] = block_wise
            func_dict['GNC'] = gaussian_noise_color
            func_dict['GB'] = gaussian_blur
            func_dict['JPEG'] = pixelation

            return func_dict[type]

        def get_distortion_parameter(type, level):
            param_dict = dict()  # a dict of list
            param_dict['BW'] = [16, 32, 48]  # larger, worse
            param_dict['GNC'] = [0.001, 0.002, 0.005]  # larger, worse
            param_dict['GB'] = [7, 9, 13]  # larger, worse
            param_dict['JPEG'] = [2, 3, 4]  # larger, worse
            # level starts from 1, list starts from 0
            return param_dict[type][level - 1]

        level = int(level)
        dist_function = get_distortion_function(type)
        dist_param = get_distortion_parameter(type, level)
        return dist_function, dist_param

# utils functions of distortions
def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr

def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr

# distortion functions
def block_wise(img, param):
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for i in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h:r_h + width, r_w:r_w + width, :] = block

    return img

def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) *
         np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    img = np.clip(b, 0, 255).astype(np.uint8)

    return img

def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img

def pixelation(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img

# ------------------------------------------------------------------------------------------------------------------- #
# for test on DFDC dataset
# ------------------------------------------------------------------------------------------------------------------- #
def set_dataset_singleGPU_DFDC(preprocess, n_frames):
    dataset = BuildDFDCdataset_online(preprocess, n_frames)
    print(f"successfully build DFDC test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return dataset, data_loader_val

import os
import pandas as pd
def init_dfdc():
    label = pd.read_csv('/data1/mianzou2/DFDCtest/labels.csv', delimiter=',')
    folder_list = [f'/data0/mian2/DFDC/faces/{i}' for i in label['filename'].tolist()]
    label_list = label['label'].tolist()

    return folder_list, label_list

class BuildDFDCdataset_online(Dataset):
    def __init__(self, preprocess, n_frames=32):
        super().__init__()

        self.n_frames = n_frames
        self.transform = preprocess

        self.folder_list, self.target_list = init_dfdc()

        self.vid_file_path = []
        self.label_list = []
        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_all_vid(self):
        for idx, vid_file_path in enumerate(self.folder_list):
            vid_file = vid_file_path.split('.')[0]
            if not os.path.exists(vid_file):
                continue
            length = len(os.listdir(vid_file))
            if length <= self.n_frames:
                continue
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)

            label = self.target_list[idx]

            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)


# ------------------------------------------------------------------------------------------------------------------- #
# for test on DF-1.0
# ------------------------------------------------------------------------------------------------------------------- #
def set_dataset_singleGPU_Deeper(preprocess, datapath='/data0/mian2/DeeperForensics/',
                                 n_frames=32, mode='end_to_end'):
    dataset = BuildDeeperDataset_online(preprocess,  datapath, n_frames, mode)
    print(f"successfully build DeeperForensics-1.0 test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return dataset, data_loader_val

# for DeeperForensics-1.0 dataset
class BuildDeeperDataset_online(Dataset):
    def __init__(self, preprocess, datapath='/data/DF-1.0/', n_frames=32, mode='end_to_end'):
        super().__init__()

        self.n_frames = n_frames
        self.datapath_manip = os.path.join(datapath, 'faces', mode)
        self.datapath_real = os.path.join('/data0/mian2/FF++/', 'original_sequences/youtube', 'c23', 'faces')
        self.transform = preprocess

        self.vid_file_path = []
        self.label_list = []

        # for real path
        img_lines, _, _ = self.get_name_candi()
        idx = 1
        for file in os.listdir(self.datapath_real):
            label = 0
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_real, file)
            self.vid_file_path.append((filepath, label))
            idx = idx + 1

        # for fake path
        fake_tmp = []
        idx = 1
        for file in os.listdir(self.datapath_manip):
            label = 1
            filepath = os.path.join(self.datapath_manip, file)
            fake_tmp.append((filepath, label))
            idx = idx + 1

        # pdb.set_trace()
        fake = fake_tmp
        # fake = random.sample(fake_tmp, len(self.vid_file_path))
        self.vid_file_path.extend(fake)


        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_name_candi(self):
        with open("/data1/mianzou2/dataset-ffpp/test.json", 'r') as fd:
            data = json.load(fd)
            img_lines = []
            fake_lines = []
            real_lines = []
            for pair in data:
                r1, r2 = pair
                img_lines.append('{}'.format(r1))
                img_lines.append('{}'.format(r2))
                img_lines.append('{}_{}'.format(r1, r2))
                img_lines.append('{}_{}'.format(r2, r1))

                real_lines.append('{}'.format(r1))
                real_lines.append('{}'.format(r2))
                fake_lines.append('{}_{}'.format(r1, r2))
                fake_lines.append('{}_{}'.format(r2, r1))
        return img_lines, real_lines, fake_lines

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)


# ------------------------------------------------------------------------------------------------------------------- #
# for FaceShifter dataset
class BuildFShDataset_online(Dataset):
    def __init__(self, preprocess, datapath = '/data0/mian2/FaceShifter/', n_frames=32, mode='c23'):
        super().__init__()

        self.n_frames = n_frames
        self.datapath_manip = os.path.join(datapath, mode, 'faces')
        self.datapath_real = os.path.join('/data0/mian2/FF++/', 'original_sequences/youtube', 'c23', 'faces')
        self.transform = preprocess

        self.vid_file_path = []
        self.label_list = []

        # for real path
        img_lines, _, _ = self.get_name_candi()
        for file in os.listdir(self.datapath_real):
            label = 0
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_real, file)
            self.vid_file_path.append((filepath, label))

        # for fake path
        for file in os.listdir(self.datapath_manip):
            label = 1
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_manip, file)
            self.vid_file_path.append((filepath, label))

        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_name_candi(self):
        with open("/data1/mianzou2/dataset-ffpp/test.json", 'r') as fd:
            data = json.load(fd)
            img_lines = []
            fake_lines = []
            real_lines = []
            for pair in data:
                r1, r2 = pair
                img_lines.append('{}'.format(r1))
                img_lines.append('{}'.format(r2))
                img_lines.append('{}_{}'.format(r1, r2))
                img_lines.append('{}_{}'.format(r2, r1))

                real_lines.append('{}'.format(r1))
                real_lines.append('{}'.format(r2))
                fake_lines.append('{}_{}'.format(r1, r2))
                fake_lines.append('{}_{}'.format(r2, r1))
        return img_lines, real_lines, fake_lines

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)

def set_dataset_singleGPU_FSh(preprocess, datapath = '/data/FSh', n_frames=32, mode='c23'):
    dataset = BuildFShDataset_online(preprocess,  datapath, n_frames, mode)
    print(f"successfully build FaceShifter test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    return dataset, data_loader_val

class MyDataset_FFSC(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None,
                 aug_probs=1.0, output_addr=False):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue

            label = int(words[1][0])
            imgs.append((words[0], label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.output_addr = output_addr
        self.aug_probs = aug_probs


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if random.random() < self.aug_probs:
            img = self.perturbation_jpeg(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.output_addr:
            return img, label, fn

        return img, label

    def __len__(self):
        return len(self.imgs)

    def perturbation_jpeg(self, im):
        distortion = ['BW', 'GNC', 'GB', 'JPEG', 'RealJPEG']
        type = random.choice(distortion)
        # if random.random() < 0.6:
        #     level = random.randint(1,2)
        # else:
        #     level = 3
        if random.random() < 0.5:
            level = 1
        else:
            level = 2

        if type != 'RealJPEG':
            im = np.asarray(im)
            im = np.copy(im)
            im = np.flip(im, 2)

            dist_function, dist_param = self.get_perturb(type, level)
            im = dist_function(im, dist_param)
            im = Image.fromarray(np.flip(im, 2))

            return im
        else: # REAL JPEG
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)  # PIL转cv2
            # pdb.set_trace()
            # # quality = random.randint(75, 100)
            # quality = random.choice([75, 80, 90])
            quality = random.choice([70, 80, 90])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', im, encode_param)[1]
            im = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
            im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            return im

    def get_perturb(self, type, level):
        def get_distortion_function(type):
            func_dict = dict()  # a dict of function
            func_dict['BW'] = block_wise
            func_dict['GNC'] = gaussian_noise_color
            func_dict['GB'] = gaussian_blur
            func_dict['JPEG'] = pixelation

            return func_dict[type]

        def get_distortion_parameter(type, level):
            param_dict = dict()  # a dict of list
            param_dict['BW'] = [16, 32, 48]  # larger, worse
            param_dict['GNC'] = [0.001, 0.002, 0.005]  # larger, worse
            param_dict['GB'] = [7, 9, 13]  # larger, worse
            param_dict['JPEG'] = [2, 3, 4]  # larger, worse
            # level starts from 1, list starts from 0
            return param_dict[type][level - 1]

        level = int(level)
        dist_function = get_distortion_function(type)
        dist_param = get_distortion_parameter(type, level)
        return dist_function, dist_param

# ------------------------------------------------------------------------------------------------------------------- #
# for Celeb-DF dataset
class BuildCelebDFdataset_online(Dataset):
    def __init__(self, preprocess, datapath = '/data/CDF/faces/', n_frames=32):
        super().__init__()

        self.n_frames = n_frames

        self.datapath = datapath
        self.vid_file_path = []
        self.label_list = []

        d_type = ['RealCDF', 'FakeCDF']
        for type in d_type:
            if type == 'RealCDF':
                label = 0
            elif type == 'FakeCDF':
                label = 1
            dir_path = os.path.join(self.datapath, type)
            for file in os.listdir(dir_path):
                filepath = os.path.join(dir_path, file)
                self.vid_file_path.append((filepath, label))

        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

        self.transform = preprocess

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)

def set_dataset_singleGPU_CDF(preprocess, datapath, n_frames):
    dataset = BuildCelebDFdataset_online(preprocess,  datapath, n_frames)
    print(f"successfully build CDF test dataset")

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return dataset, data_loader_val