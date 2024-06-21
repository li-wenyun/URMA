import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os
from scipy.special import comb

from attacked_model import Attacked_Model
from models import SurrogateImageModel, SurrogateTextModel, PerturbationGenerator, PerturbationSupervisor, ImageGenerator, Discriminator, GANLoss,SurrogateModel
from utils import mkdir_p, calc_hamming, CalcMap, image_normalization, image_restoration, return_results,get_batch ,cal_Precision_Recall_Curve,CalcTopMap,CalcTopMapWithPR
import scipy.io as sio
import open_clip

def get_label(dataset,num_label):
    num_data = len(dataset)
    L = torch.zeros(num_data, num_label)
    for i, data in enumerate(dataset):
        L[i, :] = data[1].unsqueeze(0).data
    return L

# def clamp(delta, clean_imgs):
#     MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
#     STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
#
#     clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
#     clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data
#
#     return clamp_delta

def clamp(delta, clean_imgs):

    clamp_imgs = (delta.data + clean_imgs.data).clamp(0, 1)
    clamp_delta = clamp_imgs - clean_imgs.data

    return clamp_delta

class URMA(nn.Module):
    def __init__(self, args):
        super(EQB2A, self).__init__()
        self.args = args
        self._build_model()
        self._save_setting()
    
    def _build_model(self):
        self.attacked_model = Attacked_Model(self.args.method, self.args.dataset, self.args.bit, self.args.attacked_models_path, self.args.dataset_path)
        self.attacked_model.eval().cuda()
        
        self.SM=SurrogateModel().train().cuda()
        self.criterionGAN = GANLoss().cuda()

    def _save_setting(self):
        self.output = os.path.join(self.args.output_path, '{}_{}_{}_{}'.format(self.args.output_dir, self.args.method, self.args.dataset,self.args.bit))
        self.model_dir = os.path.join(self.output, 'Model')
        # self.image_dir = os.path.join(self.output, 'Image')
        mkdir_p(self.model_dir)
        # mkdir_p(self.image_dir)

    def save_surrogate_model(self):
        torch.save(self.SM.state_dict(), os.path.join(self.model_dir, 'surrogate.pth'))

    def load_surrogate_model(self):
        self.SM.load_state_dict(torch.load(os.path.join(self.model_dir, 'surrogate.pth')))
        self.SM.eval().cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def untarget_adv(self,  query, query1,
                   epsilon=0.03125, alpha=3/255, num_iter=700):

        delta = torch.zeros_like(query,requires_grad=True)
        query_code=self.SM(query)
        query1_code=self.SM(query1)
        clean_output = calc_hamming(query_code, query1_code) / self.args.bit
        one=torch.zeros_like(clean_output)
        alienation_loss = torch.nn.MarginRankingLoss(margin=0.1)
        for i in range(num_iter):
            self.SM.zero_grad()
            adversarial_code=self.SM(query+delta)
            adversarial_output=calc_hamming(query_code,adversarial_code)
            loss=alienation_loss(clean_output,adversarial_output,one)

            # delta.retain_grad()
            loss.backward(retain_graph=True)
            # print(
            #     'x requires grad: {},  is leaf: {},  grad: {},  grad_fn: {}.'
            #     .format(delta.requires_grad, delta.is_leaf, delta.grad, delta.grad_fn)
            # )
            delta.data = delta + alpha * delta.grad.detach().sign()
            # delta.data = delta + alpha *delta.grad.detach() / torch.norm(delta.grad.detach(), 2)

            # delta.data = clamp(delta, query).clamp(-epsilon, epsilon)
            delta.data =clamp(delta, query).clamp(-epsilon, epsilon)
            delta.grad.zero_()

        return delta.detach()
        # x_adv = query.detach().clone()
        # x_adv.requires_grad = True
        # alienation_loss = torch.nn.MarginRankingLoss(margin=0.4)
        # clean_output = self.SM(query, query1)
        # one = torch.ones_like(clean_output)
        # for i in range(num_iter):
        #     self.SM.zero_grad()
        #     adversarial_output = self.SM(x_adv, query1)
        #     loss=alienation_loss(clean_output,adversarial_output,one)
        #     # loss=adversarial_output.mean()
        #     loss.backward(retain_graph=True)
        #     # print(torch.autograd.grad(loss, x_adv, create_graph=True,allow_unused=True))
        #     grad = x_adv.grad.detach()
        #     grad = grad.sign()
        #     x_adv = x_adv +alpha*grad
        # x_adv=query+ torch.clamp(x_adv - query, min=-epsilon, max=epsilon)
        # x_adv = x_adv.detach()
        # x_adv = torch.clamp(x_adv, *self.clamp)
        # return x_adv



    def test_attacked_model(self, val_set, database_set):
        print('test attacked model...')
        IqB = self.attacked_model.generate_image_hashcode(val_set)
        IdB = self.attacked_model.generate_image_hashcode(database_set)
        LqB=self.attacked_model.generate_label(val_set)
        LdB=self.attacked_model.generate_label(database_set)
        # I2I_map = CalcMap(IqB, IdB, LqB, LdB, 10)
        # I2I_map=CalcTopMap(IqB.numpy(), IdB.numpy(), LqB.numpy(), LdB.numpy(), 10)
        mAP, cum_prec, cum_recall=CalcTopMapWithPR(IqB.numpy(), LqB.numpy(), IdB.numpy(), LdB.numpy(), 10)
        num_dataset=IdB.shape[0]
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]
        # r, p=cal_Precision_Recall_Curve(IqB.numpy(), IdB.numpy(), LqB.numpy(), LdB.numpy())
        sio.savemat(os.path.join(self.model_dir, 'clean_code.mat'),{'index': index,
                                                                    "P": c_prec,
                                                                    "R": c_recall
                                                                    })
        print('I2I@50: {:.4f}'.format(mAP))

    def train_knockoff(self, train_set):
        print('train knockoff...')
        query_sampling_number = 2000
        near_sample_number = 5
        rank_sample_number = 5
        optimizer= torch.optim.Adam(self.SM.parameters(), lr=self.args.kilr, betas=(0.5, 0.999))
        index_FS_I = np.random.choice(range(len(train_set)), query_sampling_number, replace = False)
        # print(index_FS_I[0])
        qIB = self.attacked_model.generate_image_hashcode(torch.utils.data.Subset(train_set, index_FS_I))
        dTB = self.attacked_model.generate_image_hashcode(train_set)
        index_matrix_before_IT = return_results(index_FS_I, qIB, dTB, near_sample_number, rank_sample_number)
        train_sample_numbers = query_sampling_number * comb(rank_sample_number, 2).astype(int)
        index_matrix_after_IT = np.zeros((train_sample_numbers, 4), int)
        line = 0
        for i in range(query_sampling_number):
            for j in range(near_sample_number+1, near_sample_number+rank_sample_number): #取出rank_sample_number个数据
                for k in range(j+1, near_sample_number+rank_sample_number+1): #k记录检索距离
                    index_matrix_after_IT[line, :3] = index_matrix_before_IT[i, [0, j, k]]
                    index_matrix_after_IT[line, 3] = k-j
                    line = line + 1
        ranking_loss = torch.nn.MarginRankingLoss(margin=0.05)
        for epoch in range(self.args.ke):
            index = np.random.permutation(train_sample_numbers)
            for i in range(train_sample_numbers // self.args.kbz + 1):
                # optimizer_STM.zero_grad()
                optimizer.zero_grad()
                # optimizer_SIM.zero_grad()
                end_index = min((i+1)*self.args.kbz, train_sample_numbers)
                num_index = end_index - i*self.args.kbz
                ind = index[i*self.args.kbz : end_index]
                # print(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 0])).shape)
                anchor=self.SM(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 0])))
                rank1=self.SM(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 1])))
                rank2=self.SM(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 2])))
                hamming_rank1 = calc_hamming(anchor, rank1) / self.args.bit
                hamming_rank2 = calc_hamming(anchor, rank2) / self.args.bit
                # rank1_IT=self.SM(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 0])),
                #                  get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 1])))
                # rank2_IT=self.SM(get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 0])),
                #                  get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 2])))
                ranking_target_IT =  1. / torch.from_numpy(index_matrix_after_IT[ind, 3]).type(torch.float).cuda() # ranking_target_IT = - torch.ones(num_index).cuda()

                rank_loss_IT = ranking_loss(hamming_rank1, hamming_rank2, ranking_target_IT)
                
                rank_loss_IT.backward()
                optimizer.step()
                # optimizer_SIM.step()
            print('epoch:{:2d}      rank_loss:{:.4f} '
                .format(epoch,  rank_loss_IT))
        self.save_surrogate_model()
        print('train knockoff done.')

    def test_knockoff(self, val_set, database_set, k=None):
        print('test surrogate model...')
        self.load_surrogate_model()
        Lq=get_label(val_set,self.attacked_model.num_label)
        Ld=get_label(database_set,self.attacked_model.num_label)
        iq=[]
        id=[]
        for data in val_set:
            output = self.SM(data[0].type(torch.float).unsqueeze(0).cuda())
            iq.append(output.cpu().data)
        for data in database_set:
            output = self.SM(data[0].type(torch.float).unsqueeze(0).cuda())
            id.append(output.cpu().data)   
        Iq=torch.cat(iq)
        Id=torch.cat(id)
        # print(Id.shape)
        mAP, cum_prec, cum_recall=CalcTopMapWithPR(Iq.numpy(), Lq.numpy(), Id.numpy(), Ld.numpy(), 10)
        num_dataset=Id.shape[0]
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]
        sio.savemat(os.path.join(self.model_dir, 'knockoff_code.mat'),{'index': index,
                                                                    "P": c_prec,
                                                                    "R": c_recall
                                                                    })
        # map = CalcMap(Iq, Id, Lq, Ld, 10)
        print('I2T@50: {:.4f}'.format(mAP))

    def calc_dist(self,model, val_data, database_set):
        num = len(database_set)
        result = torch.zeros(num).cuda()
        # print(database_set[0][0].unsqueeze(0).shape)
        # print(val_data.unsqueeze(0).shape)
        for i in range(num):
            result[i] =model(val_data.unsqueeze(0).cuda(),database_set[i][0].unsqueeze(0).cuda())
        return result

    def train_attack_model(self, train_set, database_set):
        print('train attack model...')
        query_sampling_numbers = 2400
        near_sample_number = 5
        index_FS_I = np.random.choice(range(len(train_set)), query_sampling_numbers, replace = False)
        # QI = self.attacked_model.generate_image_hashcode(torch.utils.data.Subset(train_set, index_FS_I))
        QIB = self.attacked_model.generate_image_hashcode(torch.utils.data.Subset(train_set, index_FS_I))
        IB = self.attacked_model.generate_image_hashcode(train_set)
        index_matrix_after_IT = return_results(index_FS_I, QIB, IB, near_sample_number, 0)

        self.load_surrogate_model()


        # for epoch in range(self.args.ae):
        attacked_image = []

        reconstruction = []
        index = np.random.permutation(query_sampling_numbers)
        for i in range(query_sampling_numbers // self.args.abz + 1):
            end_index = min((i+1)*self.args.abz, query_sampling_numbers)
                # num_index = end_index - i*self.args.abz
            ind = index[i*self.args.abz : end_index]
            if len(ind)>0:
                batch_image = get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 0]))
                returned_image = get_batch(torch.utils.data.Subset(train_set, index_matrix_after_IT[ind, 1]))
                delta=self.untarget_adv(batch_image,returned_image)
                reconstruction.append(delta.mean())
                # attacked_image.append(
                #     self.attacked_model.generate_image_hashcode((delta + batch_image)).cpu().detach().numpy())
                attacked_image.append(self.attacked_model.generate_image_feature(delta + batch_image).cpu().detach())

            # map=self.test_knockoff(attacked_image,clean_image)
        attacked_image=torch.cat(attacked_image,0)
            # attacked_image=torch.from_numpy(attacked_code)
        reconstruction=torch.tensor(reconstruction)
        recon=torch.dist(reconstruction,torch.zeros_like(reconstruction),p=2)
        IdB=self.attacked_model.generate_image_hashcode(database_set)
        print('    reconstruction:{:.4f}  '
                .format(  recon))
        Lq = get_label(torch.utils.data.Subset(train_set, index_FS_I),
                           num_label=self.attacked_model.num_label)
        Ld = get_label(database_set,self.attacked_model.num_label)
        mAP, cum_prec, cum_recall=CalcTopMapWithPR(attacked_image.numpy(), Lq.numpy(), IdB.numpy(), Ld.numpy(), 10)
        num_dataset=IdB.shape[0]
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]
        # map=CalcMap(attacked_image,IdB,Lq,Ld,10)
        # r, p = cal_Precision_Recall_Curve(attacked_image.numpy(),IdB.numpy(),Lq.numpy(),Ld.numpy())
        sio.savemat(os.path.join(self.model_dir, 'adv_code.mat'), {'index': index,
                                                                    "P": c_prec,
                                                                    "R": c_recall
                                                                    })
        print('train attack model done.')
        print('I2I@50: {:.4f}'.format(mAP))

    # def test_attack_model(self, attacked_image, database_set,Lq,Ld):
    #     print('test surrogate model...')
    #     IqB = self.attacked_model.generate_image_hashcode(attacked_image)
    #     IdB = self.attacked_model.generate_image_hashcode(database_set)
    #     I2I_map = CalcMap(IqB, IdB, Lq,Ld, 50)
    #
    #     print('I2I@50: {:.4f}'.format(I2I_map))

