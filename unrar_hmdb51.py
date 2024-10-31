import os
import patoolib


# src: path for hmdb51_org.rar
# dest_dir: directory to store all extracted folders of HMDB51
def unrar_dataset(src, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok = False)
    patoolib.extract_archive(src, outdir=dest_dir)
    for rar_f in os.listdir(dest_dir):
        if rar_f.endswith('.rar'):
            rar_path = os.path.join(dest_dir, rar_f)
            patoolib.extract_archive(rar_path, outdir=dest_dir)
            os.remove(rar_path)


# src: path for hmdb51_org.rar
# dest_dir: directory to store all extracted folders of HMDB51
def unrar_dataset_update(src, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=False)
    
    # Extract the main .rar file
    patoolib.extract_archive(src, outdir=dest_dir)
    
    # Iterate through the extracted folder structure in dest_dir
    for root, dirs, files in os.walk(dest_dir):
        for rar_f in files:
            if rar_f.endswith('.rar'):
                rar_path = os.path.join(root, rar_f)
                patoolib.extract_archive(rar_path, outdir=root)
                os.remove(rar_path)  # Remove the rar file after extraction


if __name__=="__main__":
    src_path = '/Users/jialinli//Desktop/GraduateSchool/Fall2024/EN705_643_DeepLearning_Pytorch/Video_Classification/datasets/src_data/hmdb51_org.rar'
    dest_path = '/Users/jialinli//Desktop/GraduateSchool/Fall2024/EN705_643_DeepLearning_Pytorch/Video_Classification/datasets/raw_data/HMDB51'
    # unrar_dataset(src_path, dest_path)
    unrar_dataset_update(src_path, dest_path)    