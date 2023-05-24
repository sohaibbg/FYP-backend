import os


def get_directory_path(file_path):
    directory_path = os.path.dirname(file_path)
    return directory_path


if __name__ == '__main__':
    file_path = "assets\images\\vishal.png"
    print(get_directory_path(file_path=file_path))
