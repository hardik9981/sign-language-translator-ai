import os
import shutil

source_folder = "/Users/apple/Downloads/images/train"   # train folder path
target_folder = "dataset"

# classes detect karne ke liye
def get_class_name(filename):
    return filename.split(".")[0]

for file in os.listdir(source_folder):
    if file.endswith(".jpg"):
        class_name = file.split(".")[0]

        class_folder = os.path.join(target_folder, class_name)

        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        src = os.path.join(source_folder, file)
        dst = os.path.join(class_folder, file)

        shutil.copy(src, dst)

print("Dataset organized successfully")