import torch
import torch.nn as nn
from .sko.GA import GA
from .sko.DE import DE
from ..utils import *
from ..attack import Attack
from types import MethodType, FunctionType
import warnings
import sys
import multiprocessing

# please refer to the official code for the installation of the package
# https://github.com/zhaoshiji123/LPM

import warnings 

class LPM(Attack):
    """
     LPM (Learnable Patch-wise Masks)
    'Boosting Adversarial Transferability with Learnable Patch-wise Masks (IEEE MM 2023)'(https://ieeexplore.ieee.org/abstract/document/10251606)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/lpm/resnet18 --attack lpm --model=resnet18 --batchsize 1
        python main.py --input_dir ./path/to/data --output_dir adv_data/lpm/resnet18 --eval
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,  targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='lmp', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.HEIGHT = 224
        self.WIDTH = 224
        self.maxiter = 10   
        self.patch_size = 32
        self.popsize = 40
        self.b_s = 20
        simulated_names = ['resnet50','vgg16','densenet161']
        self.gray_models = self.load_models(simulated_names)
        warnings.warn(" Please refer to the official code for the installation of the sko package: https://github.com/zhaoshiji123/LP")
        
    def load_models(seld, model_names):
        # load single model
        def load_single_model(model_name):
            if model_name in models.__dict__.keys():
                print('=> Loading model {} from torchvision.models'.format(model_name))
                model = models.__dict__[model_name](weights="DEFAULT")
            elif model_name in timm.list_models():
                print('=> Loading model {} from timm.models'.format(model_name))
                model = timm.create_model(model_name, pretrained=True)
            else:
                raise ValueError('Model {} not supported'.format(model_name))
            return wrap_model(model.eval().cuda())
        
        models_list = []
        for model_name in model_names:
            models_list.append(load_single_model(model_name))
        return models_list
    

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        if(data.shape[0]>1):
            raise ValueError("Please set the batch_size==1!")
        
        data = data[0]
        label = label[0]

        bounds = [(0,1)] * int(self.WIDTH/self.patch_size) * int(self.HEIGHT/self.patch_size)
        def myfunc(x, img=data, label=label):
            
            return self.predict_transfer_score(x, img, label, self.model, self.gray_models, self.b_s)

        def callback_fn(x, convergence):
            return True
        
        lb = [0] * len(bounds)
        ub = [elem[1] for elem in bounds]
        # import pdb;pdb.set_trace()
        de = MyDE(func=myfunc, n_dim=len(bounds), size_pop=self.popsize, max_iter=self.maxiter, prob_mut=0.001, lb=lb, ub=ub, precision=1, img=None, label=None)

        masks, y = de.run()
        
        mask = torch.from_numpy(masks)
        mask = mask.reshape(-1, int(self.HEIGHT/self.patch_size), int(self.WIDTH/self.patch_size))
        
        delta = self.batch_attack_final_multiple_mask_2(data, mask, label, self.model, M_num=12, pop_size=self.popsize)  # TODO

        return delta.detach()

    def batch_attack_final_multiple_mask_2(self, img, mask, label, white_models, M_num=4, pop_size=20):
        mask_final = torch.zeros([mask.shape[0], mask.shape[1]*self.patch_size, mask.shape[2]*self.patch_size], dtype=torch.int)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]*self.patch_size):
                for k in range(mask.shape[2]):
                    mask_final[i][j][k*self.patch_size:k*self.patch_size + self.patch_size] = mask[i][int(j/self.patch_size)][int(k)]
        # mask_final = mask_final[:,:299,:299]
        mask = mask_final[:,None,:,:]
        mask = torch.cat((mask,mask,mask),1)
        mask = mask.float()

        mask = mask.cuda()
        
        X_ori = torch.stack([img])
        X = X_ori

        labels = torch.stack([label])
        X = X.cuda()
        labels = labels.cuda()
        delta = torch.zeros_like(X, requires_grad=True).cuda()
        grad_momentum = 0
        # M_num = int(mask.shape[0]/8)
        cnt = 0
        # M_num = 4
        for t in range(10):
            g_temp = []
            for tt in range(M_num):
                # if args.input_diversity:  # use Input Diversity
                X_adv = X + delta
                X_adv[:,:,:224,:224] = X_adv[:,:,:224,:224] * mask[cnt%pop_size]
                cnt += 1
                ensemble_logits = self.get_logits(X_adv, white_models)

                # Calculate the loss
                loss = self.get_loss(ensemble_logits, labels)

                # Calculate the gradients
                grad = self.get_grad(loss, delta)

                g_temp.append(grad)
            # calculate the mean and cancel out the noise, retained the effective noise
            g = 0.0
            for j in range(M_num):      
                g += g_temp[j]
            grad_momentum = self.get_momentum(g, grad_momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, X, grad_momentum, self.alpha)
        # X_adv = X_ori + delta
        return delta.detach()

    def get_logits(self, X_adv, model):
        return model(X_adv)
    
    def score_transferability(self, X_adv, label, gray_models):
        labels = label
        sum_score = np.zeros((len(gray_models), X_adv.shape[0]))
        model_num = 0
        with torch.no_grad():
            for model in gray_models:
                logits = self.get_logits(X_adv, model)
                loss = -nn.CrossEntropyLoss(reduce=False)(logits, labels)
                sum_score[model_num] += loss.detach().cpu().numpy()
                model_num += 1
            Var0 = sum_score.var(axis = 0)
            Mean0 = sum_score.mean(axis = 0)
        final_sumscore =  Var0 + Mean0
        return final_sumscore
    
    def batch_attack(self, img, mask, labels, white_models):
        mask_final = torch.zeros([mask.shape[0], mask.shape[1]*self.patch_size, mask.shape[2]*self.patch_size], dtype=torch.int)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]*self.patch_size):
                for k in range(mask.shape[2]):
                    mask_final[i][j][k*self.patch_size:k*self.patch_size + self.patch_size] = mask[i][int(j/self.patch_size)][int(k)]
        mask = mask_final[:,None,:,:]
        mask = torch.cat((mask,mask,mask),1)

        # assert False
        mask = mask.float()
        mask = mask.cuda()
        X_ori = torch.squeeze(img)
        X = X_ori.clone().cuda()

        labels = labels.cuda()
        delta = torch.zeros_like(X, requires_grad=True).cuda()
        grad_momentum = 0
        for t in range(10):
            X_adv = X + delta
            X_adv[:,:,:224,:224] = X_adv[:,:,:224,:224] * mask
            ensemble_logits = self.get_logits(X_adv, white_models)

            # Calculate the loss
            loss = self.get_loss(ensemble_logits, labels)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Ti
            # grad = F.conv2d(grad, TI_kernel(kernel_size=5, nsig=3), bias=None, stride=1, padding=(2,2), groups=3)
            # Mi
            grad_momentum = self.get_momentum(grad, grad_momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, X, grad_momentum, self.alpha)

        X_adv = X_ori + delta.detach()
        return X_adv

    def predict_transfer_score(self, x, img, label, white_models, gray_models, batch_size=4):
        # 每个个体的得分，通过每个mask单独作用到图像进行对抗攻击产生对抗样本在一组黑盒模型上的效果得分获得
        mask = torch.from_numpy(x)
        mask = mask.reshape(-1, int(self.HEIGHT/self.patch_size), int(self.WIDTH/self.patch_size))
        numsum = x.shape[0]
        scorelist = []
        bn = int(np.ceil(numsum/batch_size))
        for i in range(bn):
            bs = i*batch_size
            be = min((i+1)*batch_size, numsum)
            bn = be-bs
            X_adv = self.batch_attack(torch.stack([img]*bn), mask[bs:be], torch.stack([label]*bn), white_models)
            scorelist = np.append(scorelist, self.score_transferability(X_adv, torch.stack([label]*bn), gray_models))
        return scorelist

class MyDE(GA):
    # 可自定义排序，杂交，变异，选择
    def ranking(self):
        # import pdb;pdb.set_trace()
        self.Chrom = self.Chrom[np.argsort(self.Y),:]
        self.Y = self.Y[(np.argsort(self.Y))]

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        generation_best_index = self.Y.argmin()
        best_chrom = self.Chrom[generation_best_index]
        best_chrom_Y = self.Y[generation_best_index]
        scale_inbreeding = 0.3
        cross_chrom_size = int(scale_inbreeding * self.size_pop)
        # print(cross_chrom_size)
        superior_size = int(0.3 * self.size_pop)
        generation_superior = self.Chrom[:superior_size,:]
        # half_size_pop = int(size_pop / 2)
        # Chrom1, Chrom2 = self.Chrom[:size_pop,:][:half_size_pop], self.Chrom[:size_pop,:][half_size_pop:]
        self.crossover_Chrom = np.zeros(shape=(cross_chrom_size, len_chrom), dtype=int)
        # print(self.crossover_Chrom.shape)
        for i in range(cross_chrom_size):
            n1 = np.random.randint(0, superior_size, 2)
            # print(n1.shape)
            while n1[0] == n1[1]:
                n1 = np.random.randint(0, superior_size, 2)
            # 让 0 跟多一些
            check_1 = 1
            check_2 = 0
            for j in range(self.len_chrom):
                if generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = 1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = 0
                elif generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_1]][j]
                    check_1 = 1 - check_1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_2]][j]
                    check_2 = 1 - check_2
        return self.crossover_Chrom

 
    def mutation(self):
        scale_inbreeding = 0.3 #+ self.iter/self.max_iter*(0.8-0.2)
        rate = 0.1
        middle_1 = np.zeros((int(self.size_pop*(1-scale_inbreeding)), int(rate * self.len_chrom)))
        middle_2 = np.ones((int(self.size_pop*(1-scale_inbreeding)),self.len_chrom - int(rate * self.len_chrom)))
        self.mutation_Chrom = np.concatenate((middle_1,middle_2), axis=1)
        for i in range(self.mutation_Chrom.shape[0]):
            self.mutation_Chrom[i] = np.random.permutation(self.mutation_Chrom[i])
        return self.mutation_Chrom


    def selection(self, tourn_size=3):
        '''
        greedy selection
        '''
        # 上一代个体Chrom,得分self.Y 
        # 得到这一代个体以及分数
        offspring_Chrom = np.vstack((self.crossover_Chrom,self.mutation_Chrom))
        f_offspring  = self.func(offspring_Chrom)
        # f_chrom = self.Y.copy()
        # print("this generate score:")
        # print(f_offspring)
        num_inbreeding = int(0.3 * self.size_pop)
        selection_chrom = np.vstack((offspring_Chrom, self.Chrom))
        selection_chrom_Y = np.hstack((f_offspring, self.Y))
        # print(selection_chrom_Y)
        generation_best_index = selection_chrom_Y.argmin()
        # print(selection_chrom[generation_best_index])
        
        
        a, indices = np.unique(selection_chrom_Y, return_index=True)
        # print(a)
        # print(indices)
        
        selection_chrom_1 = np.zeros_like(selection_chrom[0:len(a)])
        selection_chrom_1 = selection_chrom[indices]
        # selection_chrom = selection_chrom[np.argsort(selection_chrom_Y),:]
        # selection_chrom_Y = selection_chrom_Y[(np.argsort(selection_chrom_Y))]
        # print("selection_chrom_1")
        # print(selection_chrom_1)
        if len(a) >= self.size_pop:
            self.Chrom = selection_chrom_1[:self.size_pop,:]
            self.Y = a[:self.size_pop]
        else:
            self.Chrom[0: len(a)] = selection_chrom_1[:len(a),:]
            self.Y[0: len(a)] = a[:len(a)]
            self.Chrom[len(a):self.size_pop] = selection_chrom_1[len(a)-1]
            self.Y[len(a):self.size_pop] = a[len(a)-1]
        # print(self.Chrom[0])
        # assert False
