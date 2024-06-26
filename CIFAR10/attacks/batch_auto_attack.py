import torch
from .attack import Attack
from common.log import log
from .auto_attack import AutoAttack


class BatchAutoAttack(Attack):
    """
    Random sampling.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchAutoAttack, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.epsilon = None
        """ (float) Epsilon. """

        self.version = 'standard'
        """ (str) Version. """

        self.norm = 'Linf'
        """ (str) Norm. """

    def run(self, model, images, objective, writer=None, prefix=''):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param images: images
        :type images: torch.autograd.Variable
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        :param writer: summary writer
        :type writer: common.summary.SummaryWriter
        :param prefix: prefix for writer
        :type prefix: str
        """

        assert self.epsilon is not None
        assert self.version in ['standard', 'plus', 'rand']

        super(BatchAutoAttack, self).run(model, images, objective, prefix)

        if not images.is_contiguous():
            images = images.contiguous()

        batch_size = images.size(0)
        #logits = model(images)
        #print(objective.success(logits))
        adversary = AutoAttack(model, norm=self.norm, eps=self.epsilon, log_path=None, version=self.version, verbose=False)
        #images = images.permute(0, 2, 3, 1)
        adv_complete = adversary.run_standard_evaluation(images, objective.true_classes, bs=batch_size)
        self.perturbations = adv_complete - images

        logits = model(images + self.perturbations)
        errors = objective(logits)
        log('success: %g' % (torch.sum(objective.success(logits)).item()/float(batch_size)))

        success_perturbations = self.perturbations.detach().cpu().numpy()
        success_errors = errors.detach().cpu().numpy()
        return success_perturbations, success_errors