import os
# Danh sách các lớp/tên thư mục

# Đường dẫn thư mục chứa dữ liệu
folder_path = './new_noise_data/'

# Duyệt qua từng lớp/thư mục
count = 501                     #Set the begin index

# Duyệt qua từng tệp trong thư mục hiện tại
for filename in os.listdir(folder_path):

    basename, extension = os.path.splitext(filename)

    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, str(count)+'.jpg')
    
    # Đổi tên tệp
    os.rename(old_path, new_path)
    
    count += 1

    print("Đổi tên các file thành công!")