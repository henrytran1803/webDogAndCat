import os
import random
from shutil import copyfile

def remove_files(paths, folder):
    for path in paths:
        file_path = os.path.join(folder, path)
        try:
            os.remove(file_path)
            print(f"Đã xóa: {file_path}")
        except FileNotFoundError:
            print(f"Không tìm thấy tệp: {file_path}")
        except Exception as e:
            print(f"Lỗi khi xóa tệp {file_path}: {e}")

def create_train_val_dirs(root_path):
    os.mkdir(root_path)
    train_dir = os.path.join(root_path, 'training')
    os.mkdir(train_dir)
    val_dir = os.path.join(root_path, 'validation')
    os.mkdir(val_dir)

    cat_train_dir = os.path.join(train_dir, 'cats')
    os.mkdir(cat_train_dir)
    dog_train_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(dog_train_dir)

    cat_val_dir = os.path.join(val_dir, 'cats')
    os.mkdir(cat_val_dir)
    dog_val_dir = os.path.join(val_dir, 'dogs')
    os.mkdir(dog_val_dir)

def split_data(source, training, validation, testing, split_size_train, split_size_val, split_size_test):
    files = [filename for filename in os.listdir(source) if os.path.getsize(os.path.join(source, filename)) > 0]

    training_length = int(len(files) * split_size_train)
    validation_length = int(len(files) * split_size_val)
    testing_length = int(len(files) * split_size_test)

    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    validation_set = shuffled_set[training_length:(training_length + validation_length)]
    testing_set = shuffled_set[(training_length + validation_length):]

    def copy_files(file_set, src, dest):
        for filename in file_set:
            source_file = os.path.join(src, filename)
            destination_file = os.path.join(dest, filename)
            copyfile(source_file, destination_file)

    copy_files(training_set, source, training)
    copy_files(validation_set, source, validation)
    copy_files(testing_set, source, testing)

# Define paths
CAT_SOURCE_DIR = "model/PetImages/Cat/"
DOG_SOURCE_DIR = "model/PetImages/Dog/"
TRAINING_DIR = "model/working/cats-v-dogs/training/"
VALIDATION_DIR = "model/working/cats-v-dogs/validation/"
TEST_DIR = "model/working/cats-v-dogs/test/"

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")
TEST_CATS_DIR = os.path.join(TEST_DIR, "cats/")
TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")
TEST_DOGS_DIR = os.path.join(TEST_DIR, "dogs/")

# Empty directories
def empty_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif len(os.listdir(directory)) > 0:
        for file in os.scandir(directory):
            os.remove(file.path)

empty_directory(TRAINING_CATS_DIR)
empty_directory(TRAINING_DOGS_DIR)
empty_directory(VALIDATION_CATS_DIR)
empty_directory(VALIDATION_DOGS_DIR)
empty_directory(TEST_CATS_DIR)
empty_directory(TEST_DOGS_DIR)

# Đường dẫn bạn muốn loại bỏ
paths_to_remove = [
    'Dog/Thumbs.db',
    'Cat/Thumbs.db',
    'Cat/666.jpg',
    'Dog/11702.jpg'
]

# Xóa các tệp không mong muốn
remove_files(paths_to_remove, 'model/PetImages')

# Tạo thư mục train và validation
create_train_val_dirs('model/working/cats-v-dogs')

# Chia dữ liệu thành tập train, validation, và test
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, TEST_CATS_DIR, 0.8, 0.1, 0.1)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, TEST_DOGS_DIR, 0.8, 0.1, 0.1)

# Kiểm tra số lượng ảnh trong các thư mục
print(f"\n\nThư mục gốc của mèo có {len(os.listdir(CAT_SOURCE_DIR))} ảnh")
print(f"Thư mục gốc của chó có {len(os.listdir(DOG_SOURCE_DIR))} ảnh\n")
print(f"Số lượng ảnh mèo cho tập huấn luyện: {len(os.listdir(TRAINING_CATS_DIR))}")
print(f"Số lượng ảnh chó cho tập huấn luyện: {len(os.listdir(TRAINING_DOGS_DIR))}")
print(f"Số lượng ảnh mèo cho tập validation: {len(os.listdir(VALIDATION_CATS_DIR))}")
print(f"Số lượng ảnh chó cho tập validation: {len(os.listdir(VALIDATION_DOGS_DIR))}")
print(f"Số lượng ảnh mèo cho tập test: {len(os.listdir(TEST_CATS_DIR))}")
print(f"Số lượng ảnh chó cho tập test: {len(os.listdir(TEST_DOGS_DIR))}")
