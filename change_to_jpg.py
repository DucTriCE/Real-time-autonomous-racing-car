import os

def rename_images_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Thư mục '{directory_path}' không tồn tại.")
        return

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]

    if not image_files:
        print(f"Không có tệp ảnh JPG nào trong thư mục '{directory_path}'.")
        return

    for i, image_file in enumerate(image_files):
        new_name = f"{i}.jpg"  # Đổi tên thành '0.jpg', '1.jpg', ...
        old_path = os.path.join(directory_path, image_file)
        new_path = os.path.join(directory_path, new_name)
        os.rename(old_path, new_path)
        print(f"Đổi tên '{image_file}' thành '{new_name}'.")

# Sử dụng hàm
directory_path = "./sign_more_data"
rename_images_in_directory(directory_path)