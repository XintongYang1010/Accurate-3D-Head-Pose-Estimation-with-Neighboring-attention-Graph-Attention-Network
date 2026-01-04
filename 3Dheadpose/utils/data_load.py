import torch
import numpy as np
from torch.utils.data import Dataset
import  os

def load_face_data(args):
    root = os.path.join(args.root,"dataset",args.dataset_name)

    if os.path.exists(root + "sample_all.npy"):
        all = np.load(root + "sample_all.npy", allow_pickle=True)
    else: all = None

    if args.use_augment:
        sample = np.load(root + 'sample_aug.npy', allow_pickle=True)
        label = np.load(root + 'label_aug.npy', allow_pickle=True)
        if args.use_heatmap:
            heat_map= np.load(root + 'heat map_aug.npy', allow_pickle=True)
            return sample, label, heat_map,all
        else:
            return sample, label,all
    else:
        sample = np.load(root + '/sample.npy', allow_pickle=True)
        label = np.load(root + '/label.npy', allow_pickle=True)
        filename =np.load(root+'/filename.npy', allow_pickle=True)
        if args.use_heatmap:
            heat_map = np.load(root + 'heatmap.npy', allow_pickle=True)
            return sample,label,heat_map,all,filename
        else:
            return sample,label,all,filename


class FaceLandmarkData(Dataset):
    def __init__(self,args,partition='trainval'):
        self.heat=[]
        if args.use_heatmap:
            self.data, self.label, self.seg,self.all_sample,self.filename = load_face_data(args)
        else:
            self.data,self.label,self.all_sample,self.filename=load_face_data(args)
        self.partition = partition
        self.root=args.root
        self.model_name =args.model_name

    def __getitem__(self, item):

        if self.heat!=[]:
            sample_T, label_T, heat_T,= torch.Tensor(self.data), torch.Tensor(self.label), torch.Tensor(self.seg)
            heatmap = heat_T[item]
        else:
            sample_T, label_T = torch.Tensor(self.data), torch.Tensor(self.label)
        face = sample_T[item]
        landmark = label_T[item]
        file = self.filename[item]

        if self.partition == 'trainval':
            # 打乱点云中点的顺序
            indices = list(range(face.size()[0]))
            np.random.shuffle(indices)
            face = face[indices]
            if self.heat!=[]:
                heatmap = heatmap[indices]
                return face, landmark, heatmap,file
            else:
                return face, landmark,torch.zeros(0),file
        else:
            all_sample_T = torch.Tensor(self.all_sample)
            all_sample = all_sample_T[item]
            if self.heat!=[]:
                return face, landmark, heatmap, all_sample,file

            else:
                return face, landmark,torch.zeros(0), all_sample,file

    def __len__(self):
        return np.array(self.data).shape[0]





