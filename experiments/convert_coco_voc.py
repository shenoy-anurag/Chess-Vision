classes = [
    {
        "id": 0,
        "name": "pieces",
        "supercategory": "none"
    },
    {
        "id": 1,
        "name": "bishop",
        "supercategory": "pieces"
    },
    {
        "id": 2,
        "name": "black-bishop",
        "supercategory": "pieces"
    },
    {
        "id": 3,
        "name": "black-king",
        "supercategory": "pieces"
    },
    {
        "id": 4,
        "name": "black-knight",
        "supercategory": "pieces"
    },
    {
        "id": 5,
        "name": "black-pawn",
        "supercategory": "pieces"
    },
    {
        "id": 6,
        "name": "black-queen",
        "supercategory": "pieces"
    },
    {
        "id": 7,
        "name": "black-rook",
        "supercategory": "pieces"
    },
    {
        "id": 8,
        "name": "white-bishop",
        "supercategory": "pieces"
    },
    {
        "id": 9,
        "name": "white-king",
        "supercategory": "pieces"
    },
    {
        "id": 10,
        "name": "white-knight",
        "supercategory": "pieces"
    },
    {
        "id": 11,
        "name": "white-pawn",
        "supercategory": "pieces"
    },
    {
        "id": 12,
        "name": "white-queen",
        "supercategory": "pieces"
    },
    {
        "id": 13,
        "name": "white-rook",
        "supercategory": "pieces"
    }
]

class_index_to_name = {}

for c in classes:
    class_index_to_name[c["id"]] = c["name"]

print(class_index_to_name)