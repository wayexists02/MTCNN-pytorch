import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import pickle
import cv2

from env import *


class DataLoader():

    CELEBA_DATA_PATH = "data/celeba"
    WIDER_DATA_PATH = "data/wider"

    def __init__(self, batch_size, train):
        # self.nettype = nettype
        self.batch_size = batch_size
        self.train = train

        # if self.nettype == "pnet":
        #     self.size = 12
        # elif self.nettype == "rnet":
        #     self.size = 24
        # elif self.nettype == "onet":
        #     self.size = 48
        # else:
        #     raise ValueError("INVALID net type!")

        self.size = 48

        # self.celeba_train, self.celeba_valid, self.celeba_test = self._read_celeba_list()
        self.celeba_train, self.celeba_valid = self._read_celeba_list()
        self.wider_train, self.wider_valid = self._read_wider_list()

        if train is True:
            del self.celeba_valid, self.wider_valid

            n = len(self.wider_train)
            self.n_batches = int(np.ceil(n / batch_size))

            self.celeba_path = os.path.join(DataLoader.CELEBA_DATA_PATH, "train").replace("\\", "/")
            self.wider_path = os.path.join(DataLoader.WIDER_DATA_PATH, "train").replace("\\", "/")

        else:
            del self.celeba_train, self.wider_train

            n = len(self.wider_valid)
            self.n_batches = int(np.ceil(n / batch_size))

            self.celeba_path = os.path.join(DataLoader.CELEBA_DATA_PATH, "valid").replace("\\", "/")
            self.wider_path = os.path.join(DataLoader.WIDER_DATA_PATH, "valid").replace("\\", "/")

    def __len__(self):
        if hasattr(self, "n_batches"):
            return self.n_batches
        else:
            raise AttributeError("DataLoader has no attribute named 'n_batches'.")

    def next_batch(self):
        if self.train is True:
            celeba = self.celeba_train
            wider = self.wider_train
        else:
            celeba = self.celeba_valid
            wider = self.wider_valid

        n = len(wider)

        for b in range(self.n_batches):
            start = b * self.batch_size
            end = min((b + 1)*self.batch_size, n)

            img_batch = np.zeros((end - start, 3, self.size, self.size))
            face_batch = np.zeros((end - start,))
            bbox_batch = np.zeros((end - start, 4))
            lm_batch = np.zeros((end - start, 10))

            for i in range((end - start) // 2):
                while True:
                    ind = np.random.randint(len(wider))
                    wider_sample = wider[ind]
                    negative_sample = self._get_negative_sample(wider_sample)
                    if negative_sample is not None:
                        break

                img_batch[i] = negative_sample
                face_batch[i] = 0

            for i in range((end - start) // 2, end - start):
                while True:
                    ind = np.random.randint(len(celeba))
                    celeba_sample = celeba[ind]
                    positive_sample, bbox, lm = self._get_positive_sample(celeba_sample)
                    if positive_sample is not None:
                        break

                img_batch[i] = positive_sample
                face_batch[i] = 1
                bbox_batch[i] = bbox
                lm_batch[i] = lm

            yield img_batch, face_batch, bbox_batch, lm_batch

    def _read_celeba_list(self):
        datasets = []

        # 이미 데이터셋들이 덤프되어 있는 경우, 로드해온다.
        if os.path.exists("data/celeba/celeba_train.bin") and os.path.exists("data/celeba/celeba_valid.bin"): # and os.path.exists("data/celeba_test.bin"):
            with open("data/celeba/celeba_train.bin", "rb") as f:
                train = pickle.loads(f.read())
            
            with open("data/celeba/celeba_valid.bin", "rb") as f:
                valid = pickle.loads(f.read())
            
            # with open("data/celeba_test.bin", "rb") as f:
            #     test = pickle.loads(f.read())

            # datasets.extend([train, valid, test])
            datasets.extend([train, valid])
            return datasets

        # 덤프되어 있지 않은 경우, 만든다.
        for cat in ["train", "valid"]:
            dataset = []

            # train/valid/test attribute 파일 읽어옴
            bbox_file_path = os.path.join(DataLoader.CELEBA_DATA_PATH, f"{cat}_bbox.csv").replace("\\", "/")
            lm_file_path = os.path.join(DataLoader.CELEBA_DATA_PATH, f"{cat}_lm.csv").replace("\\", "/")

            bbox_file = pd.read_csv(bbox_file_path)
            lm_file = pd.read_csv(lm_file_path)

            assert len(bbox_file) == len(lm_file), "len(bbox_file) != len(lm_file)"
            n = len(bbox_file)

            # 모든 파일을 돌면서 데이터셋을 만듦
            for i in range(n):
                bbox = bbox_file.iloc[i][["x_1", "y_1", "width", "height"]]
                lm = lm_file.iloc[i][["lefteye_x", "lefteye_y", "righteye_x", "righteye_y", "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y"]]

                dataset.append(
                    (
                        str(bbox_file.iloc[i]["image_id"]),
                        list(map(float, bbox.values)),
                        list(map(float, lm.values))
                    )
                )

                # if (i + 1) % 1000 == 0:
                #     print(f"{i+1}")

            datasets.append(dataset)

            # 만든 데이터셋을 덤프
            with open(f"data/celeba/celeba_{cat}.bin", "wb") as f:
                dump_str = pickle.dumps(dataset)
                f.write(dump_str)

            print(f"Completed for celeba {cat} data.")

        return datasets

    def _read_wider_list(self):
        datasets = []
        
        for cat in ["train", "valid"]:
            with open(f"data/wider/wider_{cat}.bin", "rb") as f:
                dataset = pickle.loads(f.read())
                datasets.append(dataset)

        return datasets

    def _get_negative_sample(self, wider_sample):
        filename, bboxes = wider_sample
        
        path = os.path.join(self.wider_path, filename).replace("\\", "/")
        img = cv2.imread(path)

        h, w, c = img.shape

        for x, y in zip(np.random.randint(0, w - self.size, size=10), np.random.randint(0, h - self.size, size=10)):
            box = [x, y, x+self.size, y+self.size]
            
            maximum_iou = self._compute_iou(box, bboxes)
            if maximum_iou < self.size*self.size*0.2:
                imgslice = img[y:y+self.size, x:x+self.size]

                imgslice = cv2.cvtColor(imgslice, cv2.COLOR_BGR2RGB)

                # transpose
                imgslice = imgslice.transpose(2, 0, 1)

                # standardize
                imgslice = imgslice.astype(np.float32)
                imgslice = (imgslice - 128) / 256

                # noising
                if np.random.rand() < 0.4:
                    noise = np.random.randn(*imgslice.shape) * 0.05
                    imgslice += noise

                return imgslice

        return None

    def _get_positive_sample(self, celeba_sample):
        filename, bbox, lms = celeba_sample

        path = os.path.join(self.celeba_path, filename).replace("\\", "/")
        img = cv2.imread(path)

        cut_point = min(bbox[2], bbox[3]) // 3

        if cut_point == 0:
            return None, None, None

        cut_x = max(0, np.random.randint(bbox[0] - cut_point, bbox[0] + cut_point))
        cut_y = max(0, np.random.randint(bbox[1] - cut_point, bbox[1] + cut_point))
        cut_size = np.random.randint(min(bbox[2], bbox[3]) - cut_point, min(bbox[2], bbox[3]) + cut_point)

        if cut_x + cut_size > img.shape[1]:
            cut_size = img.shape[1] - cut_x
        if cut_y + cut_size > img.shape[0]:
            cut_size = img.shape[0] - cut_y

        img = img[
            cut_y:cut_y + cut_size,
            cut_x:cut_x + cut_size
        ]

        normed_bbox = [
            float(bbox[0] - cut_x) / cut_size,
            float(bbox[1] - cut_y) / cut_size,
            float(bbox[2]) / cut_size,
            float(bbox[3]) / cut_size
        ]
        
        normed_lms = []
        for i in range(5):
            x, y = lms[2*i:2*(i+1)]
            
            normed_lms.append(float(x - cut_x) / cut_size)
            normed_lms.append(float(y - cut_y) / cut_size)

        img = cv2.resize(img, dsize=(self.size, self.size))
        
        # bgr2rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # transpose
        img = img.transpose(2, 0, 1)

        # standardize
        img = img.astype(np.float32)
        img = (img - 128) / 256

        # noising
        if np.random.rand() < 0.4:
            noise = np.random.randn(*img.shape) * 0.05
            img += noise

        return img, normed_bbox, normed_lms

    def _compute_iou(self, box, bboxes):
        if len(bboxes) == 0:
            return 0

        union_areas = []
        
        for bbox in bboxes:
            union_area = 0

            union_width = min(box[0] + box[2] - bbox[0], bbox[0] + bbox[2] - box[0])
            union_height = min(box[1] + box[3] - bbox[1], bbox[1] + bbox[3] - box[1])

            if union_width < 0 or union_height < 0:
                union_area = 0
            
            else:
                union_area = union_width * union_height

            union_areas.append(union_area)

        return max(union_areas)
