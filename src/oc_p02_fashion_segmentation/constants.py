CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17,
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

CLOTHING_CLASSES = {
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
}

ACCESSORY_CLASSES = {
    "Hat",
    "Sunglasses",
    "Belt",
    "Bag",
    "Scarf",
}

SHOE_CLASSES = {
    "Left-shoe",
    "Right-shoe",
}

CLOTHING_CLASS_IDS = sorted(CLASS_MAPPING[name] for name in CLOTHING_CLASSES)
ACCESSORY_CLASS_IDS = sorted(CLASS_MAPPING[name] for name in ACCESSORY_CLASSES)
SHOE_CLASS_IDS = sorted(CLASS_MAPPING[name] for name in SHOE_CLASSES)
