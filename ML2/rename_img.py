import os

def rename_images(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    for index, image in enumerate(images):
        new_name = f"temp_{index}.jpg"  # Temporary name to avoid conflicts
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    
    # Rename temp files to final names
    for index in range(len(images)):
        temp_name = os.path.join(folder_path, f"temp_{index}.jpg")
        final_name = os.path.join(folder_path, f"{index}.jpg")
        os.rename(temp_name, final_name)
    
    print("Renaming complete of ", folder_path)



DATA_DIR = './data5'
for dir_ in os.listdir(DATA_DIR):
    rename_images(os.path.join(DATA_DIR, dir_))
