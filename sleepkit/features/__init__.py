from .featset01 import FeatSet01
from .featset02 import FeatSet02
from .featset03 import FeatSet03
from .featset04 import FeatSet04
from .featset05 import FeatSet05
from .featset06 import FeatSet06
from .featureset import SKFeatureSet
from .store import FeatureStore

FeatureStore.register(FeatSet01.name(), FeatSet01)
FeatureStore.register(FeatSet02.name(), FeatSet02)
FeatureStore.register(FeatSet03.name(), FeatSet03)
FeatureStore.register(FeatSet04.name(), FeatSet04)
FeatureStore.register(FeatSet05.name(), FeatSet05)
FeatureStore.register(FeatSet06.name(), FeatSet06)

from .generate import generate_feature_set  # pylint: disable=wrong-import-position
