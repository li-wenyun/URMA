import torch
import torch.nn as nn
import scipy.io as scio
import torchvision

class Attacked_Model(nn.Module):
    def __init__(self, method, dataset, bit, attacked_models_path, dataset_path):
        super(Attacked_Model, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit
        vgg_path = dataset_path + 'imagenet-vgg-f.mat'
        if self.dataset == 'isic':
            self.num_label = 8
        if self.dataset == 'kvasir':
            self.num_label = 8
        if self.dataset == 'Pattern':
            self.num_label = 38
        if self.dataset == 'derment':
            self.num_label = 23
        

        if self.method == 'HashNet':
            path = attacked_models_path + str(self.method) + '_' + self.dataset  + '/{}.pth'.format(self.bit)
            from attacked_methods.HashNet.HashNet import ResNet
            self.model=ResNet(self.bit)
            self.model.cuda().float()
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            # pretrain_model = scio.loadmat(vgg_path)
            # self.image_hashing_model = ImgModule(self.bit, pretrain_model)
            # self.text_hashing_model = TxtModule(tag_dim, self.bit)
            # self.image_hashing_model.load(load_img_path)
            # self.text_hashing_model.load(load_txt_path)
            # self.image_hashing_model.cuda().eval()
            # self.text_hashing_model.cuda().eval()
        if self.method == 'CSQ':
            path = attacked_models_path + str(self.method) + '_' + self.dataset  + '/{}.pth'.format(self.bit)
            from attacked_methods.CSQ.CSQ import ResNet
            self.model = ResNet(self.bit)
            self.model.cuda().float()
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        if self.method == 'DPSH':
            path = attacked_models_path + str(self.method) + '_' + self.dataset  + '/{}.pth'.format(self.bit)
            from attacked_methods.CSQ.CSQ import ResNet
            self.model = ResNet(self.bit)
            self.model.cuda().float()
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        



    def generate_image_feature(self, data):
        if self.method == 'CSQ':
            output = self.model(data)
            B = output.cpu().data
        if self.method == 'HashNet':
            output= self.model(data)
            B = output.cpu().data
        if self.method == 'DPSH':
            output= self.model(data)
            B=output.cpu().data
        return torch.sign(B)
    
    
    def generate_label(self,dataset):
        num_data=len(dataset)
        L=torch.zeros(num_data,self.num_label)
        # print(L.shape)
        for i, data in enumerate(dataset):
            L[i,:]=data[1].unsqueeze(0).data
        return L
    

    def generate_image_hashcode(self, dataset):
        num_data = len(dataset)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'CSQ':
            for i, data in enumerate(dataset):
                output = self.model(data[0].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DHCNN':
            for i, data in enumerate(dataset):
                output, _ = self.model(data[0].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'HashNet':
            for i, data in enumerate(dataset):
                output=self.model(data[0].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'DPSH':
            for i, data in enumerate(dataset):
                output=self.model(data[0].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        return torch.sign(B)

    

    
