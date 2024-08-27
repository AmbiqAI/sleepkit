"""
# Features API

This module provides a set of feature sets for sleep analysis tasks.

Feature sets are defined as classes that inherit from `FeatureSet` abstract class.
Several built-in feature sets are provided in this module via the `FeatureFactory` class.

Classes:
    FeatureSet: Base class for all feature sets.
    FS_W_PA_14: Feature set with 14 features generated from PPG and IMU captured on wrist.
    FS_C_EAR_9: Feature set with 9 features generated from ECG, RESP, and ACC captured on chest.
    FS_W_A_5: Feature set with 5 features generated from IMU captured on wrist.
    FS_H_E_10: Feature set with 10 features generated from EEG/EOG captured on head.
    FS_W_P_5: Feature set with 5 features generated from PPG captured on wrist.
    FS_W_P_40: Feature set with 40 features generated from PPG captured on wrist.

"""

from .fs_w_pa_14 import FS_W_PA_14
from .fs_c_ear_9 import FS_C_EAR_9
from .fs_w_a_5 import FS_W_A_5
from .fs_h_e_10 import FS_H_E_10
from .fs_w_p_5 import FS_W_P_5
from .fs_w_p_40 import FS_W_P_40
from .featureset import FeatureSet
from .h5dataloader import H5Dataloader

import neuralspot_edge as nse

FeatureFactory = nse.utils.ItemFactory[FeatureSet].shared("SKFeatureFactory")

FeatureFactory.register(FS_W_PA_14.name(), FS_W_PA_14)
FeatureFactory.register(FS_C_EAR_9.name(), FS_C_EAR_9)
FeatureFactory.register(FS_W_A_5.name(), FS_W_A_5)
FeatureFactory.register(FS_H_E_10.name(), FS_H_E_10)
FeatureFactory.register(FS_W_P_5.name(), FS_W_P_5)
FeatureFactory.register(FS_W_P_40.name(), FS_W_P_40)
