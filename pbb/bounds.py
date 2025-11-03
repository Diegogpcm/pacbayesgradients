import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F


class PBBobj():
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired 
    training objective and evaluate the risk certificate at the end of training. 

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)
    
    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem
    
    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective
    
    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    mc_samples : int
        number of Monte Carlo samples when estimating the risk

    kl_penalty : float
        penalty for the kl coefficient in the training objective
    
    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, objective='fquad', pmin=1e-4, classes=10, delta=0.025,
    delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda', n_posterior=30000, n_bound=30000):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        self.runavg_sample_01_loss = 0.9
        self.runavg_sample_xe_loss = 0.2
        
        


    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded == True:
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
            
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well
        outputs = net(data, sample=True,
                      clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(
            outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(
            target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1-(correct/total)
              
        # for _acc targets, hacky solution
        #if '_acc' in self.objective:
        self.runavg_sample_01_loss = 0.99*self.runavg_sample_01_loss + 0.01*loss_01
        try:
            self.runavg_sample_xe_loss = 0.99*self.runavg_sample_xe_loss + 0.01*loss_ce.item()
        except:
             self.runavg_sample_xe_loss = 0.99*self.runavg_sample_xe_loss + 0.01*loss_ce
        
        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, train_size, lambda_var=None):
        # compute training objectives
        if self.objective == 'fnew':
            kl = kl * self.kl_penalty
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            repeated_term = repeated_kl_ratio*(1-empirical_risk) 
            first_term = torch.sqrt(
                empirical_risk + repeated_term)
            second_term = torch.sqrt(repeated_term)
            train_obj = torch.pow(first_term + second_term, 2)
            
        elif self.objective == 'fquad':
            kl = kl * self.kl_penalty
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            first_term = torch.sqrt(
                empirical_risk + repeated_kl_ratio)
            second_term = torch.sqrt(repeated_kl_ratio)
            train_obj = torch.pow(first_term + second_term, 2)
            
        elif self.objective == 'f_rts':
            kl = kl * self.kl_penalty
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            train_obj = empirical_risk + repeated_kl_ratio + torch.sqrt(2*empirical_risk*repeated_kl_ratio)
            
        elif self.objective == 'fgrad':
            kl = kl * self.kl_penalty
            K = torch.div(kl + np.log((2*np.sqrt(train_size))/self.delta), train_size)
            try:
                current_loss = empirical_risk.item()
            except:
                current_loss = empirical_risk
            current_bound = inv_kl(current_loss, K)
            
            # to prevent extreme case errors
            if (current_bound == 0) or (current_bound == 1):
                print(f'current bound: {current_bound}, current loss: {current_loss}, current K: {K}')
                current_bound = np.abs(current_bound - 1e-5)
            
            dkl_db = ((1-current_loss)/(1-current_bound) - current_loss/current_bound) #d. parcial de kl(x,b) resp. b
            dkl_dx = np.log(current_loss/current_bound) - np.log((1-current_loss)/(1-current_bound)) #d. parcial de kl(x,b) resp. x
            train_obj = -(dkl_dx*empirical_risk - K)/dkl_db
            
        elif self.objective == 'fquad_acc':
            kl = kl * self.kl_penalty
                
            try:
                current_xe_loss = empirical_risk.item()
            except:
                current_xe_loss = empirical_risk
            
            current_01_loss = self.runavg_sample_01_loss
            empirical_risk = empirical_risk * (current_01_loss / self.runavg_sample_xe_loss)
            
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            first_term = torch.sqrt(
                empirical_risk + repeated_kl_ratio)
            second_term = torch.sqrt(repeated_kl_ratio)
            train_obj = torch.pow(first_term + second_term, 2)
            
        elif self.objective == 'f_rts_acc':
            kl = kl * self.kl_penalty
                
            try:
                current_xe_loss = empirical_risk.item()
            except:
                current_xe_loss = empirical_risk
            
            current_01_loss = self.runavg_sample_01_loss
            empirical_risk = empirical_risk * (current_01_loss / self.runavg_sample_xe_loss)
            
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            train_obj = empirical_risk + repeated_kl_ratio + torch.sqrt(2*empirical_risk*repeated_kl_ratio)
            
        elif self.objective == 'fgrad_acc':
            kl = kl * self.kl_penalty
            K = torch.div(kl + np.log((2*np.sqrt(train_size))/self.delta), train_size)
            
            try:
                current_xe_loss = empirical_risk.item()
            except:
                current_xe_loss = empirical_risk
            
            current_01_loss = self.runavg_sample_01_loss
            current_01_bound = inv_kl(current_01_loss, K)
            
            empirical_risk = empirical_risk * (current_01_loss / self.runavg_sample_xe_loss)
            
            if (current_01_bound == 0) or (current_01_bound == 1):
                print(f'current bound: {current_01_bound}, current loss: {current_01_bound}, current K: {K}')
                current_01_bound = np.abs(current_01_bound - 1e-5)
            
            dkl_db = ((1-current_01_loss)/(1-current_01_bound) - current_01_loss/current_01_bound) #d. parcial de kl(x,b) resp. b
            dkl_dx = np.log(current_01_loss/current_01_bound) - np.log((1-current_01_loss)/(1-current_01_bound)) #d. parcial de kl(x,b) resp. x
            train_obj = -(dkl_dx*empirical_risk - K)/dkl_db
            
        elif self.objective == 'flamb':
            kl = kl * self.kl_penalty
            lamb = lambda_var.lamb_scaled
            kl_term = torch.div(
                kl + np.log((2*np.sqrt(train_size)) / self.delta), train_size*lamb*(1 - lamb/2))
            first_term = torch.div(empirical_risk, 1 - lamb/2)
            train_obj = first_term + kl_term
            
        elif self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
            
        elif self.objective == 'bbb':

            train_obj = empirical_risk + \
                self.kl_penalty * (kl/train_size)
            
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
            
        return train_obj


    def mcsampling(self, net, input, target, batches=False, clamping=True, data_loader=None):
        # compute empirical risk with Monte Carlo sampling
        error = 0.0
        cross_entropy = 0.0
        if batches:
            for batch_id, (data_batch, target_batch) in enumerate(tqdm(data_loader)):
                data_batch, target_batch = data_batch.to(
                    self.device), target_batch.to(self.device)
                cross_entropy_mc = 0.0
                error_mc = 0.0
                for i in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(net,
                                                              data_batch, target_batch, clamping)
                    cross_entropy_mc += loss_ce
                    error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
                cross_entropy += cross_entropy_mc/self.mc_samples
                error += error_mc/self.mc_samples
            # we average cross-entropy and 0-1 error over all batches
            cross_entropy /= batch_id
            error /= batch_id
        else:
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for i in trange(self.mc_samples):
                loss_ce, loss_01, _ = self.compute_losses(net,
                                                          input, target, clamping)
                cross_entropy_mc += loss_ce
                error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc/self.mc_samples
            error += error_mc/self.mc_samples
        return cross_entropy, error

    def train_obj(self, net, input, target, clamping=True, lambda_var=None):
        # compute train objective and return all metrics
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net,
                                                        input, target, clamping)

        train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var)
        return train_obj, kl/self.n_posterior, outputs, loss_ce, loss_01

    def compute_final_stats_risk(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        # compute all final stats and risk certificates

        kl = net.compute_kl()
        if data_loader:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=True,
                                                 clamping=True, data_loader=data_loader)
        else:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=False,
                                                 clamping=True)

        empirical_risk_ce = inv_kl(
            error_ce.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01 = inv_kl(
            error_01, np.log(2/self.delta_test)/self.mc_samples)

        train_obj = self.bound(empirical_risk_ce, kl, self.n_posterior, lambda_var)

        risk_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 *
                                                             np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
        risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 *
                                                             np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
        return train_obj.item(), kl.item()/self.n_bound, error_ce, error_01, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01


def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*math.log(qs/p)+0)
        else:
            ikl = ks-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
