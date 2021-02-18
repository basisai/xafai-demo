import json

# List raw numeric and categorical features
NUMERIC_FEATS = [
    'Age',
    'Capital Gain',
    'Capital Loss',
    'Hours per week',
]

CATEGORICAL_FEATS = [
    'Workclass',
    'Education',
    'Marital Status',
    'Occupation',
    'Relationship',
    'Race',
    'Sex',
    'Country',
]

# For each categorical feature, get the one-hot encoded feature names
CATEGORY_MAP = json.load(open("data/category_map.txt"))

OHE_CAT_FEATS = []
for f in CATEGORICAL_FEATS:
    OHE_CAT_FEATS.extend(CATEGORY_MAP[f])

# Train & validation features and target
FEATURES = OHE_CAT_FEATS + NUMERIC_FEATS
TARGET = "Target"
TARGET_CLASSES = [0, 1]
TARGET_NAMES = ['<=50K', '>50K']

# List privileged info
PROTECTED_FEATURES = {
    'Sex=Male': {
        'privileged_attribute_values': [1],
        'unprivileged_attribute_values': [0],
    },
    'Race=White': {
        'privileged_attribute_values': [1],
        'unprivileged_attribute_values': [0],
    },
}
