from pipeline import preprocess_imgs

preprocess = preprocess_imgs(root_path='./test_imgs')
all_file_names = preprocess.get_img_file_names()
for img_i in all_file_names:
    preprocess.get_raw_imgs_4ch(file_name=img_i)
preprocess.adjust_imgs()
preprocess.generate_mask()
embryoid_properties = preprocess.get_embryoid_properties()