import cv2
import numpy as np

import torch
from torch import nn

from models.pnet import PNet
from models.rnet import RNet
from models.onet import ONet


class Mtcnn():

    def __init__(self, max_crop=64):
        self.pnet = PNet().cuda()
        self.rnet = RNet().cuda()
        self.onet = ONet().cuda()

        self.max_crop = max_crop

    def __call__(self, x, training=False):
        if training is True:
            img, face, bbox, lm = x

            pnet_input, rnet_input, onet_input = self._get_input_tensor(img)

            # defien losses
            criterion_face = nn.NLLLoss()
            criterion_bbox = nn.MSELoss()
            criterion_lm = nn.MSELoss()

            # create torch tensors
            pnet_input = torch.FloatTensor(pnet_input).cuda()
            rnet_input = torch.FloatTensor(rnet_input).cuda()
            onet_input = torch.FloatTensor(onet_input).cuda()
            face = torch.LongTensor(face).cuda()
            bbox = torch.FloatTensor(bbox).cuda()
            lm = torch.FloatTensor(lm).cuda()

            # forward propagation
            p_face_preds, p_bbox_preds, p_lm_preds = self.pnet(pnet_input)
            r_face_preds, r_bbox_preds, r_lm_preds = self.rnet(rnet_input)
            o_face_preds, o_bbox_preds, o_lm_preds = self.onet(onet_input)

            # compute loss
            face_loss = criterion_face(p_face_preds, face) + \
                        criterion_face(r_face_preds, face) + \
                        criterion_face(o_face_preds, face)

            bbox_loss = criterion_bbox(p_bbox_preds[face == 1], bbox[face == 1]) + \
                        criterion_bbox(r_bbox_preds[face == 1], bbox[face == 1]) + \
                        criterion_bbox(o_bbox_preds[face == 1], bbox[face == 1])

            lm_loss = criterion_lm(p_lm_preds[face == 1], lm[face == 1]) + \
                      criterion_lm(r_lm_preds[face == 1], lm[face == 1]) + \
                      criterion_lm(o_lm_preds[face == 1], lm[face == 1])

            loss = face_loss + 0.5*bbox_loss + 0.5*lm_loss

            # compute accuracy
            p_equal = torch.argmax(p_face_preds, dim=1).view(*face.size()) == face
            r_equal = torch.argmax(r_face_preds, dim=1).view(*face.size()) == face
            o_equal = torch.argmax(o_face_preds, dim=1).view(*face.size()) == face

            p_face_acc = torch.mean(p_equal.type(torch.FloatTensor))
            r_face_acc = torch.mean(r_equal.type(torch.FloatTensor))
            o_face_acc = torch.mean(o_equal.type(torch.FloatTensor))

            return loss, p_face_acc, r_face_acc, o_face_acc

        else:
            with torch.no_grad():
                img = x

                # crop image
                pnet_input, rnet_input, onet_input, cache = self._crop(img, self.max_crop)

                # create torch tensors
                pnet_input = torch.FloatTensor(pnet_input.transpose(0, 3, 1, 2)).cuda()
                rnet_input = torch.FloatTensor(rnet_input.transpose(0, 3, 1, 2)).cuda()
                onet_input = torch.FloatTensor(onet_input.transpose(0, 3, 1, 2)).cuda()

                # forward prop for PNet
                p_face_preds, p_bbox_preds, p_lm_preds = self.pnet(pnet_input)
                
                rnet_input = rnet_input[torch.argmax(p_face_preds, dim=1).cpu().detach().numpy() == 1]
                onet_input = onet_input[torch.argmax(p_face_preds, dim=1).cpu().detach().numpy() == 1]
                cache = list(map(lambda elem: elem[torch.argmax(p_face_preds, dim=1).cpu().detach().numpy() == 1], cache))

                # forward prop for RNet
                r_face_preds, r_bbox_preds, r_lm_preds = self.rnet(rnet_input)
                
                onet_input = onet_input[torch.argmax(r_face_preds, dim=1).cpu().detach().numpy() == 1]
                cache = list(map(lambda elem: elem[torch.argmax(r_face_preds, dim=1).cpu().detach().numpy() == 1], cache))

                # forward prop for ONet
                o_face_preds, o_bbox_preds, o_lm_preds = self.onet(onet_input)

                return o_face_preds, o_bbox_preds, o_lm_preds, cache

    def parameters(self):
        params = []
        
        params.extend(self.pnet.parameters())
        params.extend(self.rnet.parameters())
        params.extend(self.onet.parameters())

        return params

    def save(self, path):
        state_dict = {
            "pnet": self.pnet.state_dict(),
            "rnet": self.rnet.state_dict(),
            "onet": self.onet.state_dict(),
        }

        torch.save(state_dict, path)
        print("MTCNN was saved.")

    def load(self, path):
        state_dict = torch.load(path)

        self.pnet.load_state_dict(state_dict["pnet"])
        self.rnet.load_state_dict(state_dict["rnet"])
        self.onet.load_state_dict(state_dict["onet"])

        print("MTCNN was loaded.")

    def get_coord_transformed(self, face_preds, bbox_preds, lm_preds, cache):
        bbox_preds = bbox_preds[torch.argmax(face_preds, dim=1).cpu().detach().numpy() == 1]
        lm_preds = lm_preds[torch.argmax(face_preds, dim=1).cpu().detach().numpy() == 1]
        xs = cache[0][torch.argmax(face_preds, dim=1).cpu().detach().numpy() == 1]
        ys = cache[1][torch.argmax(face_preds, dim=1).cpu().detach().numpy() == 1]
        sizes = cache[2][torch.argmax(face_preds, dim=1).cpu().detach().numpy() == 1]

        transformed = []

        for bbox, lm, x, y, size in zip(bbox_preds, lm_preds, xs, ys, sizes):
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            bbox_x = bbox_x * size + x
            bbox_y = bbox_y * size + y
            bbox_w = bbox_w * size
            bbox_h = bbox_h * size

            landmark = []
            for i in range(0, 10, 2):
                lm_x, lm_y = lm[i], lm[i+1]
                lm_x = lm_x * size + x
                lm_y = lm_y * size + y

                landmark.extend([int(lm_x), int(lm_y)])

            transformed.append([int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h), landmark])

        return transformed

    def _get_input_tensor(self, imgs):
        n, c, h, w = imgs.shape

        cvimgs = imgs.transpose(0, 2, 3, 1)

        pnet_input = np.zeros((n, c, 12, 12))
        rnet_input = np.zeros((n, c, 24, 24))
        onet_input = np.zeros((n, c, 48, 48))

        for i in range(n):
            pnet_input[i] = cv2.resize(cvimgs[i], dsize=(12, 12)).transpose(2, 0, 1)
            rnet_input[i] = cv2.resize(cvimgs[i], dsize=(24, 24)).transpose(2, 0, 1)
            onet_input[i] = cv2.resize(cvimgs[i], dsize=(48, 48)).transpose(2, 0, 1)

        return pnet_input, rnet_input, onet_input

    def _crop(self, img, max_crop):
        h, w, c = img.shape

        random_size = np.random.randint(24, min(h, w), size=max_crop)
        random_x = np.random.randint(0, min(h, w) - 24, size=max_crop)
        random_y = np.random.randint(0, min(h, w) - 24, size=max_crop)

        # filterring valid crop
        index_slice = np.ones(random_size.shape).astype(np.bool)
        index_slice = index_slice & (random_x + random_size <= w)
        index_slice = index_slice & (random_y + random_size <= h)

        random_size = random_size[index_slice]
        random_x = random_x[index_slice]
        random_y = random_y[index_slice]

        n = random_size.shape[0]

        pnet_input = np.zeros((n, 12, 12, 3))
        rnet_input = np.zeros((n, 24, 24, 3))
        onet_input = np.zeros((n, 48, 48, 3))

        i = 0

        for size, x, y in zip(random_size, random_x, random_y):
            cropped = img[y:y + size, x:x + size]
            
            pnet_input[i] = cv2.resize(cropped, dsize=(12, 12))
            rnet_input[i] = cv2.resize(cropped, dsize=(24, 24))
            onet_input[i] = cv2.resize(cropped, dsize=(48, 48))

            i += 1

        cache = [random_x, random_y, random_size]

        return pnet_input, rnet_input, onet_input, cache
