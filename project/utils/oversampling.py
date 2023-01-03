from enum import Enum
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, KMeansSMOTE
class OverSamplingMethods(Enum):
    smote = SMOTE
    smoten = SMOTEN
    smotenc = SMOTENC
    smoteenn = SMOTEENN
    kmeanssmote = KMeansSMOTE
