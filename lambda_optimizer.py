import copy
import torch

from SO_loss import pLoss, graph_SO_FFSC


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class AutoLambda_FFSC:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=0.1, dataset='ffsc'):
        self.model = model
        self.model_ = copy.deepcopy(model)
        convert_models_to_fp32(self.model_)
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks

        if dataset in ['ffsc']:
            self.pLoss = pLoss(graph_SO_FFSC())
        else:
            raise NotImplementedError
        print(f'Dataset for the lambda optimizer is {dataset}...')

    def virtual_step(self, train_x, train_y, train_mask, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        train_pred = self.model(train_x)

        train_loss = self.model_fit(train_pred, train_y, train_mask)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, train_x, train_y, train_mask,
                          val_x, val_y, val_mask,
                          alpha, alpha_lambda, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, train_mask, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y, val_mask)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y, train_mask)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                # mw.grad = - alpha * h
                mw.grad = - alpha_lambda * h

    def compute_hessian(self, d_model, train_x, train_y, train_mask):
        norm = torch.cat([w.view(-1) for w in d_model]).norm() + 1e-6
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y, train_mask)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y, train_mask)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian


    def model_fit(self, pred, gt, mask): # for FFSC probalistic loss
        total_loss, pMargin = self.pLoss(pred, gt.float(), mask, auto_mode=True)

        # for 3-level tasks
        loss_1 = total_loss[0]
        loss_2 = self.compute_one_level_loss(total_loss[1:6])
        loss_3 = self.compute_one_level_loss(total_loss[6:])
        loss = [loss_1, loss_2, loss_3]

        return loss

    def compute_one_level_loss(self, loss):
        total_loss = 0
        for loss_item in loss:
            total_loss += loss_item

        total_loss = total_loss / len(loss)
        return total_loss