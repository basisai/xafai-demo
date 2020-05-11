import json

# For explainability AI app

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
CATEGORY_MAP = json.load(open("input/category_map.txt"))

OHE_CAT_FEATS = []
for f in CATEGORICAL_FEATS:
    OHE_CAT_FEATS.extend(CATEGORY_MAP[f])

# Train & validation features and target
FEATURES = OHE_CAT_FEATS + NUMERIC_FEATS
TARGET = "Target"
TARGET_NAMES = ['<=50K', '>50K']


# For fairness AI app
# List bias and privileged info
CONFIG_FAI = {
    'Race=White': {
        'bias_info': {
            'favorable_label': 1,
            'unfavorable_label': 0,
            'protected_columns': [
                'Race=Amer-Indian-Eskimo',
                'Race=Asian-Pac-Islander',
                'Race=Black',
                'Race=Other',
                'Race=White',
            ],
        },
        'privileged_info': {
            'unprivileged_groups': [{'Race=White': 0}],
            'privileged_groups': [{'Race=White': 1}]
        },
    },
}
