import os

for root, dirs, _ in os.walk("../dataset/chess-pieces-dataset"):
    print(root, dirs, _)
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        for _, _, files in os.walk(folder_path):
            print(files)
    break

