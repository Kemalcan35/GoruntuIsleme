import cv2
import numpy as np
import json
import os

# Global değişkenler
points = []
cm_coordinates = []

# Fare ile tıklama olayı için fonksiyon
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Sadece dört nokta seçilmesine izin ver
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Yeşil nokta ekle
            cv2.imshow('Original Image', image)

# Dörtgen bulma fonksiyonu
def find_quadrilaterals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrilaterals = []

    for contour in contours:
        # Konturun alanını hesapla
        area = cv2.contourArea(contour)
        if area < 100:  # Küçük alanları atla
            continue

        # Konturu dört kenara dönüştür
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Dört kenar varsa kaydet
        if len(approx) == 4:
            quadrilaterals.append(approx)

    return quadrilaterals

# Uzaklıkları hesaplama fonksiyonu
def calculate_distances(quadrilaterals):
    distances = []
    for quadrilateral in quadrilaterals:
        # Dörtgenin merkezini hesapla
        M = cv2.moments(quadrilateral)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            # Uzaklıkları (0, 0) noktasından hesapla
            distances.append((cX, cY))  # Merkez koordinatını ekle
    return distances

# Masaüstü dizinini al
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# Masaüstünde json_output adında bir klasör oluştur
output_dir = os.path.join(desktop_path, 'json_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Resim yolunu kullanıcıdan al
image_path = input("Lütfen işlenecek resmin dosya yolunu girin: ")

# Resmi yükle
image = cv2.imread(image_path)
if image is None:
    print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# Kullanıcıdan boyutlandırma bilgisi al
width = int(input("Yeni genişlik değerini girin: "))
height = int(input("Yeni yükseklik değerini girin: "))

image = cv2.resize(image, (width, height))  # Resmi boyutlandır
cv2.imshow('Original Image', image)

# Fare olayı için pencereyi ayarla
cv2.setMouseCallback('Original Image', select_point)

# Kullanıcının tıklamasını bekle
cv2.waitKey(0)

# Dört köşe noktası (tıklanan noktalar)
if len(points) == 4:
    # Kullanıcıdan cm cinsinden koordinatları al
    for i in range(4):
        cm_x = float(input(f"{i + 1}. noktanın X koordinatını (cm cinsinden) girin: "))
        cm_y = float(input(f"{i + 1}. noktanın Y koordinatını (cm cinsinden) girin: "))
        cm_coordinates.append((cm_x, cm_y))

    pts1 = np.float32(points)  # Tıklanan dört noktayı al
    # Dönüştürülecek yeni köşe noktaları
    target_width, target_height = 280, 210  # İstediğiniz boyut
    pts2 = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])  # Düzgün bir sıralama

    # Perspektif dönüşüm matrisini hesapla
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (target_width, target_height))

    # Dönüştürülmüş görüntüde kareleri bul
    quadrilaterals = find_quadrilaterals(result)

    # Bulunan kareleri çiz ve isimlerini yazdır
    for i, quadrilateral in enumerate(quadrilaterals):
        cv2.drawContours(result, [quadrilateral], -1, (0, 255, 0), 2)  # Dörtgeni yeşil ile çiz
        # Dörtgenin merkezini hesapla
        M = cv2.moments(quadrilateral)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            # Kare ismini yazdır
            cv2.putText(result, f'Kare {i + 1}', (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # (0, 0) noktasına çizgi çiz
            cv2.line(result, (0, 0), (cX, cY), (255, 0, 0), 1)  # Mavi çizgi

    # Uzaklıkları hesapla
    distances = calculate_distances(quadrilaterals)

    # Uzaklıkları ekrana yazdır
    for i, (x, y) in enumerate(distances):
        print(f"Kare {i + 1} koordinatları (0,0) noktasına göre: X: {x}, Y: {y}")

    # Sonucu JSON formatında kaydet
    json_data = {
        "quadrilaterals": [
            {"index": i + 1, "coordinates": {"X": x, "Y": y}}
            for i, (x, y) in enumerate(distances)
        ]
    }

    json_path = os.path.join(output_dir, 'coordinates.json')
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Koordinatlar {json_path} dosyasına kaydedildi.")

    # Sonucu göster
    cv2.imshow('Warped Image with Detected Quadrilaterals', result)
else:
    print("Lütfen dört nokta seçin.")

cv2.waitKey(0)
cv2.destroyAllWindows()