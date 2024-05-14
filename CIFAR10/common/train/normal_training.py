import torch
import numpy
import common.torch
import common.summary
import common.numpy
from common.progress import ProgressBar
import numpy as np
import wandb


class NormalTraining:
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, augmentation=None, loss=common.torch.classification_loss, cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert loss is not None
        assert callable(loss)
        assert isinstance(model, torch.nn.Module)
        assert len(trainset) > 0
        assert len(testset) > 0
        assert isinstance(trainset, torch.utils.data.DataLoader)
        assert isinstance(testset, torch.utils.data.DataLoader)
        assert isinstance(trainset.sampler, torch.utils.data.RandomSampler)
        assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))


        self.progress = ProgressBar()
        """ (Timer) """

        self.model = model
        """ (torch.nn.Module) Model. """

        self.layers = range(len(list(model.parameters())))
        """ ([int]) Layers for projection. """

        self.trainset = trainset
        """ (torch.utils.data.DatLoader) Taining set. """

        self.testset = testset
        """ (torch.utils.data.DatLoader) Test set. """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (torch.optim.LRScheduler) Scheduler. """

        self.augmentation = augmentation
        """ (imgaug.augmenters.Sequential) Augmentation. """

        self.cuda = cuda
        """ (bool) Run on CUDA. """

        self.loss = loss
        """ (callable) Classificaiton loss. """

        self.train_losses = []
        self.train_partial_losses = []
        self.test_losses = []
        self.test_partial_losses = []
        self.clean_accuracies = []
        self.adv_accuracies = []

        
        
        


    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                inputs = self.augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, self.cuda)
            #inputs = inputs.permute(0, 3, 1, 2)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, self.cuda)
            assert len(list(targets.size())) == 1

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.loss(logits, targets)
            error = common.torch.classification_error(logits, targets)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

           
            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g error=%g lr=%g' % (
                loss.item(),
                error.item(),
                self.scheduler.get_lr()[0],
            ))

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.eval()
        assert self.model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        probabilities = None

        for b, (inputs, targets) in enumerate(self.testset):
            inputs = common.torch.as_variable(inputs, self.cuda)
            #inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            outputs = self.model(inputs)
            b_losses = self.loss(outputs, targets, reduction='none')
            b_errors = common.torch.classification_error(outputs, targets, reduction='none')

            losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
            errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

            self.progress('test %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                torch.mean(b_losses).item(),
                torch.mean(b_errors.float()).item()
            ))

        confidences = numpy.max(probabilities, axis=1)


        return probabilities

    def step(self, epoch, logging, train_progress_file):
        """
        Training + test step.

        :param epoch: epoch
        :type epoch: int
        :return: probabilities of test set
        :rtype: numpy.array
        """
        
        train_loss, train_partial_loss, max_grad = self.train(epoch)
        test_loss, test_partial_loss, clean_acc, adv_acc = self.test(epoch)
        

        if logging:
        
            self.train_losses.append(train_loss)
            self.train_partial_losses.append(train_partial_loss)
            self.test_losses.append(test_loss)
            self.test_partial_losses.append(test_partial_loss)
            self.clean_accuracies.append(clean_acc)
            self.adv_accuracies.append(adv_acc)
            
            
            torch.save({
            'epoch' : epoch,
            'train_losses': np.array(self.train_losses),
            'train_partial_losses': np.array(self.train_partial_losses),
            'test_losses': np.array(self.test_losses),
            'test_partial_losses': np.array(self.test_partial_losses),
            'clean_acc': np.array(self.clean_accuracies),
            'adv_acc': np.array(self.adv_accuracies),
            },train_progress_file)
            
            
            wandb.log({
                "train_loss": float(train_loss),
                "train_Clean_loss": float(train_partial_loss[0]),
                "train_Adv_loss": float(train_partial_loss[1]),
                "train_Inv_Sup_KS_loss": float(train_partial_loss[2]),
                "train_Pair_KS_loss": float(train_partial_loss[3]),
                "train_Cov_KS_loss": float(train_partial_loss[4]),
                "test_loss": float(test_loss),
                "test_Clean_loss": float(test_partial_loss[0]),
                "test_Adv_loss": float(test_partial_loss[1]),
                "test_Inv_Sup_KS_loss": float(test_partial_loss[2]),
                "test_Pair_KS_loss": float(test_partial_loss[3]),
                "test_Cov_KS_loss": float(test_partial_loss[4]),
                "clean_acc": float(clean_acc),
                "adv_acc": float(adv_acc),
                #"max_grad": float(max_grad)
                    })
            
        
        
        
        
        pass
