import cv2
import os


# dataset path
HUMAN_DATASET_PATH = 'adda/datasets/human_dataset'
ANIMAL_DATASET_PATH = 'adda/datasets/animal_face_dataset'
# cascade path in opencv
OPENCV_PATH = 'opencv-4.5.2'
FACE_CASCADE_PATH = f'{OPENCV_PATH}/data/haarcascades/haarcascade_frontalface_default.xml'
CAT_CASCADE_PATH = f'{OPENCV_PATH}/data/haarcascades/haarcascade_frontalcatface.xml'


# cut out face from original human image and save segment_image directory by image directory
# User should check that images are cut out correctly.
def human_face_segmentation_preprocess():
    original_image_data_path = os.path.join(HUMAN_DATASET_PATH, 'original')
    dataset_dir_list = os.listdir(original_image_data_path)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    for human_dir_name in dataset_dir_list:
        save_segment_image_path = f'{HUMAN_DATASET_PATH}/segment_images/{human_dir_name}'
        if not os.path.exists(save_segment_image_path):
            os.makedirs(save_segment_image_path)

        for i, image_file in enumerate(os.listdir(os.path.join(original_image_data_path, human_dir_name))):
            assert os.path.isfile(os.path.join(original_image_data_path, human_dir_name, image_file))
            image = cv2.imread(os.path.join(original_image_data_path, human_dir_name, image_file))

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_image)
            for j, [x, y, w, h] in enumerate(faces):
                face = image[y: y + h, x: x + w]
                # cv2.imwrite(f'{save_segment_image_path}/img_{i}_{j}.png', face)


# cut out face from original animal image and save segment_image directory by image directory
# User should check images which are cut out correctly.
def animal_face_segmentation_preprocess():
    original_image_data_path = os.path.join(ANIMAL_DATASET_PATH, 'original')
    dataset_dir_list = os.listdir(original_image_data_path)
    face_cascade = cv2.CascadeClassifier(CAT_CASCADE_PATH)

    for animal_dir_name in dataset_dir_list:
        save_path = f'{ANIMAL_DATASET_PATH}/segment_images/{animal_dir_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, image_file in enumerate(os.listdir(os.path.join(original_image_data_path, animal_dir_name))):
            assert os.path.isfile(os.path.join(original_image_data_path, animal_dir_name, image_file))
            image = cv2.imread(os.path.join(original_image_data_path, animal_dir_name, image_file))

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray_image)
            for j, [x, y, w, h] in enumerate(faces):
                face = image[y: y + h, x: x + w]
                # cv2.imwrite(f'{save_path}/img_{i}_{j}.png', face)


# resize segmented image and save resized directory by image directory
def resize_segmented_images(dataset_path, img_size=(128, 128)):
    dataset_list = os.listdir(f'{dataset_path}/segment_images')
    resize_exist_dir_list = os.listdir(f'{dataset_path}/resized')

    for dirs in dataset_list:
        if dirs in resize_exist_dir_list:
            continue
        save_path = f'{dataset_path}/resized/{dirs}'
        os.makedirs(save_path)
        for i, image_file in enumerate(os.listdir(os.path.join(dataset_path, dirs))):

            assert os.path.isfile(os.path.join(dataset_path, dirs, image_file)), print(f'{os.path.join(dataset_path, dirs, image_file)}')
            image = cv2.imread(os.path.join(dataset_path, dirs, image_file))
            image = cv2.resize(image, dsize=img_size, interpolation=cv2.INTER_LANCZOS4)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(f'{save_path}/img_{i}.png', gray_image)


# show normal image, reflect image and upside down image
def demonstration():
    image = cv2.imread(f'datasets/human_dataset/resized/kasumi_arimura/img_1.jpg')

    reverse_lr_image = cv2.flip(image, 0)
    reverse_ud_image = cv2.flip(image, 1)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', reverse_ud_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', reverse_lr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rename_image_file(data_path):
    dir_list = os.listdir(data_path)
    for i, dir_name in enumerate(dir_list):
        if os.path.isdir(os.path.join(data_path, dir_name)):
            file_list = os.listdir(os.path.join(data_path, dir_name))
            for j, file_name in enumerate(file_list):
                os.rename(os.path.join(data_path, dir_name, file_name), os.path.join(data_path, dir_name, f'org_img_{j}.png'))
        else:
            os.rename(os.path.join(data_path, dir_name), os.path.join(data_path, f'org_img_{i}.png'))


if __name__ == '__main__':
    # human_face_segmentation_preprocess()
    # animal_face_segmentation_preprocess()
    # resize_segmented_images(HUMAN_DATASET_PATH)
    rename_image_file(os.path.join(ANIMAL_DATASET_PATH, 'original'))
