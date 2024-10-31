import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Đường dẫn thư mục ảnh
duong_dan_thu_muc = r'C:\Users\tranm\Downloads\Panoramic radiographs with periapical lesions Dataset\Panoramic radiographs with periapical lesions Dataset\Periapical Dataset\Periapical Lesions\Original JPG Images'  # Thay bằng đường dẫn thư mục chứa ảnh nha khoa

# Đọc ảnh và nhãn từ thư mục
def doc_anh_va_nhan(duong_dan):
    anh = []
    nhan = []
    for ten_tep in os.listdir(duong_dan):
        duong_dan_tep = os.path.join(duong_dan, ten_tep)
        if os.path.isfile(duong_dan_tep):
            # Đọc ảnh và thay đổi kích thước
            img = cv2.imread(duong_dan_tep)
            img = cv2.resize(img, (64, 64))  # Thay đổi kích thước ảnh về 64x64
            img_vector = img.flatten()
            anh.append(img_vector)
            
            # Gán nhãn từ tên tệp
            try:
                # Giả sử tên tệp là 'anhX.jpg', trích xuất X là lớp
                nhan.append(int(ten_tep[3]))  # Lấy ký tự thứ 4 (đếm từ 0)
            except (ValueError, IndexError):
                print(f"Lỗi với tên tệp: {ten_tep}. Vui lòng kiểm tra định dạng.")
                nhan.append(-1)  # Gán nhãn mặc định nếu lỗi
            
    return np.array(anh), np.array(nhan)



# Đọc ảnh và nhãn
X, y = doc_anh_va_nhan(duong_dan_thu_muc)

# Chia tập dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Mô hình CART
mo_hinh_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
mo_hinh_cart.fit(X_train, y_train)
y_du_doan_cart = mo_hinh_cart.predict(X_test)
do_chinh_xac_cart = accuracy_score(y_test, y_du_doan_cart)
print("Mô hình CART - Độ chính xác:", do_chinh_xac_cart)

# Mô hình ID3
mo_hinh_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
mo_hinh_id3.fit(X_train, y_train)
y_du_doan_id3 = mo_hinh_id3.predict(X_test)
do_chinh_xac_id3 = accuracy_score(y_test, y_du_doan_id3)
print("Mô hình ID3 - Độ chính xác:", do_chinh_xac_id3)

# Hàm hiển thị ảnh đã phân lớp
def hien_thi_phan_lop(X, y_thuc, y_du_doan, tieu_de):
    so_anh_hien_thi = min(10, len(X))  # Giới hạn số ảnh hiển thị không vượt quá số ảnh có
    plt.figure(figsize=(12, 8))
    for i in range(so_anh_hien_thi):  # Hiển thị số ảnh tối đa có
        img = X[i].reshape(64, 64, 3)  # Chuyển đổi vector về dạng ảnh
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Chuyển đổi màu sắc từ BGR sang RGB
        plt.title(f"Thực tế: {y_thuc[i]}, Dự đoán: {y_du_doan[i]}")
        plt.axis('off')
    plt.suptitle(tieu_de)
    plt.show()


# Hiển thị kết quả phân lớp của mô hình CART
hien_thi_phan_lop(X_test, y_test, y_du_doan_cart, "Kết quả phân lớp CART")

# Hiển thị kết quả phân lớp của mô hình ID3
hien_thi_phan_lop(X_test, y_test, y_du_doan_id3, "Kết quả phân lớp ID3")