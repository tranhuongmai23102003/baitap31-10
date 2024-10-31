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
    anh = []  # Danh sách lưu trữ các ảnh đã được xử lý
    nhan = []  # Danh sách lưu trữ các nhãn tương ứng của ảnh
    for ten_tep in os.listdir(duong_dan):  # Duyệt qua từng tệp trong thư mục
        duong_dan_tep = os.path.join(duong_dan, ten_tep)  # Lấy đường dẫn của từng tệp
        if os.path.isfile(duong_dan_tep):  # Kiểm tra nếu là tệp
            img = cv2.imread(duong_dan_tep)  # Đọc ảnh
            img = cv2.resize(img, (64, 64))  # Thay đổi kích thước ảnh về 64x64
            img_vector = img.flatten()  # Chuyển ảnh thành vector 1D
            anh.append(img_vector)  # Thêm vector ảnh vào danh sách ảnh
            
            try:
                nhan.append(int(ten_tep[3]))  # Giả sử ký tự thứ 4 của tên là nhãn lớp, thêm vào danh sách nhãn
            except (ValueError, IndexError):
                print(f"Lỗi với tên tệp: {ten_tep}. Vui lòng kiểm tra định dạng.")
                nhan.append(-1)  # Gán nhãn mặc định là -1 nếu xảy ra lỗi
            
    return np.array(anh), np.array(nhan)  # Trả về mảng numpy của ảnh và nhãn

# Đọc ảnh và nhãn
X, y = doc_anh_va_nhan(duong_dan_thu_muc)  # Gọi hàm để lấy dữ liệu ảnh và nhãn

# Chia tập dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Chia 70% huấn luyện, 30% kiểm thử

# Mô hình CART
mo_hinh_cart = DecisionTreeClassifier(criterion='gini', random_state=42)  # Tạo mô hình cây quyết định với tiêu chí gini (CART)
mo_hinh_cart.fit(X_train, y_train)  # Huấn luyện mô hình trên tập huấn luyện
y_du_doan_cart = mo_hinh_cart.predict(X_test)  # Dự đoán trên tập kiểm thử
do_chinh_xac_cart = accuracy_score(y_test, y_du_doan_cart)  # Tính độ chính xác mô hình
print("Mô hình CART - Độ chính xác:", do_chinh_xac_cart)  # In ra độ chính xác của mô hình CART

# Mô hình ID3
mo_hinh_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)  # Tạo mô hình cây quyết định với tiêu chí entropy (ID3)
mo_hinh_id3.fit(X_train, y_train)  # Huấn luyện mô hình ID3
y_du_doan_id3 = mo_hinh_id3.predict(X_test)  # Dự đoán trên tập kiểm thử với mô hình ID3
do_chinh_xac_id3 = accuracy_score(y_test, y_du_doan_id3)  # Tính độ chính xác của mô hình ID3
print("Mô hình ID3 - Độ chính xác:", do_chinh_xac_id3)  # In ra độ chính xác của mô hình ID3

# Hàm hiển thị ảnh đã phân lớp
def hien_thi_phan_lop(X, y_thuc, y_du_doan, tieu_de):
    so_anh_hien_thi = min(10, len(X))  # Số ảnh hiển thị tối đa là 10 (nếu có ít hơn 10 thì hiển thị tất cả)
    plt.figure(figsize=(12, 8))  # Thiết lập kích thước đồ thị
    for i in range(so_anh_hien_thi):  # Lặp qua số lượng ảnh cần hiển thị
        img = X[i].reshape(64, 64, 3)  # Chuyển đổi vector ảnh 1D thành ảnh 3D 64x64x3
        plt.subplot(2, 5, i + 1)  # Tạo lưới 2x5 cho hiển thị ảnh
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Hiển thị ảnh, chuyển từ BGR sang RGB
        plt.title(f"Thực tế: {y_thuc[i]}, Dự đoán: {y_du_doan[i]}")  # Hiển thị nhãn thực tế và nhãn dự đoán
        plt.axis('off')  # Tắt các trục tọa độ
    plt.suptitle(tieu_de)  # Tiêu đề chính của hiển thị
    plt.show()  # Hiển thị các ảnh và nhãn

# Hiển thị kết quả phân lớp của mô hình CART
hien_thi_phan_lop(X_test, y_test, y_du_doan_cart, "Kết quả phân lớp CART")  # Hiển thị ảnh và nhãn với mô hình CART

# Hiển thị kết quả phân lớp của mô hình ID3
hien_thi_phan_lop(X_test, y_test, y_du_doan_id3, "Kết quả phân lớp ID3")  # Hiển thị ảnh và nhãn với mô hình ID3
