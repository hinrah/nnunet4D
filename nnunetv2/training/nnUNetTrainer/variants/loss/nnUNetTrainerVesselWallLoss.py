from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from nnunetv2.training.loss.robust_ce_loss import TopKLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, VesselWallLoss, VesselWallLossWithActivator
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import torch


class nnUNetTrainerVesselWallLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
class nnUNetTrainerVesselWallLossStronger(nnUNetTrainer):
    
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.0005)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss    

class nnUNetTrainerVesselWallLossStronger_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000    

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.0005)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss    


class nnUNetTrainerVesselWallLossEvenStronger(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.002)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss    
    
class nnUNetTrainerVesselWallLoss_1e02(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.01)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss  

    
class nnUNetTrainerVesselWallLoss_1e02_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.01)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss  

class nnUNetTrainerVesselWallLoss_1e02_8000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.01)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss  

class nnUNetTrainerVesselWallLoss_1e01(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.1)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss    
    
class nnUNetTrainerVesselWallLoss_1e01_2000epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLoss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss, dilatation_size=5, padding=2, weight_morph=0.1)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss    

class nnUNetTrainerVesselWallLossAfterHalf(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = VesselWallLossWithActivator({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.current_epoch == self.num_epochs // 2:
            self.loss.activate_vessel_wall_loss()