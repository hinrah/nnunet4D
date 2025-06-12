import shutil
from copy import deepcopy
from typing import List, Union, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

class ExperimentPlanner4D(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = True):
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """
        # Time dimension must always be first → force suppress_transpose=True
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
                    plans_name, overwrite_target_spacing, suppress_transpose=True)
        


    def determine_transpose(self):
        if self.suppress_transpose:
            return [0, 1, 2, 3], [0, 1, 2, 3]

        # todo we should use shapes for that as well. Not quite sure how yet
        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(4)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(4)]
        return transpose_forward, transpose_backward

    def plan_experiment(self):
        """
        """
        # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
       
        plan_4d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                            new_median_shape_transposed[1:],
                                                            self.generate_data_identifier('4d_fullres'),
                                                            approximate_n_voxels_dataset, _tmp)
        plan_4d_fullres['patch_size'] = tuple([np.int64(16)] + list(plan_4d_fullres['patch_size']))
        plan_4d_fullres['spacing'] = tuple([np.int64(fullres_spacing[0])] + list(plan_4d_fullres['spacing']))
        patch_size_fullres = plan_4d_fullres['patch_size']

        plan_4d_fullres['batch_dice'] = False

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # json is ###. I hate it... "Object of type int64 is not JSON serializable"
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'4d_fullres': plan_4d_fullres},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        self.plans = plans
        self.save_plans(plans)
        return plans



