
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract

def GetImage(Image):
    img = cv2.imread("../Resim/"+Image)
    return img

def filters(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gurultuazalt = cv2.bilateralFilter(img_gray, 9, 75, 75)
    histogram_e = cv2.equalizeHist(gurultuazalt)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morp_image = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel, iterations=15)
    gray_image = cv2.subtract(histogram_e, morp_image)
    ret, goruntuesikle = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    canny_goruntu = cv2.Canny(goruntuesikle, 250, 255)
    canny_goruntu = cv2.convertScaleAbs(canny_goruntu)
    cekirdek = np.ones((3, 3), np.uint8)
    gen_goruntu = cv2.dilate(canny_goruntu, cekirdek, iterations=1)
    contours, hierarchy = cv2.findContours(gen_goruntu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    final = cv2.drawContours(img, [screenCnt],-1, (9, 236, 255), 3)
    mask = np.zeros(img_gray.shape, np.uint8)
    temp_image= cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    temp_image = cv2.bitwise_and(img, img, mask=mask)
    y, cr, cb = cv2.split(cv2.cvtColor(temp_image, cv2.COLOR_RGB2YCrCb))
    y = cv2.equalizeHist(y)
    final_image = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    cv2.imwrite('Resim/end.png',final_image)
    
    plt.imshow(final_image)
    plt.axis('off')
    resize_test_license_plate = cv2.resize(
        final_image, None, fx = 2, fy = 2,
        interpolation = cv2.INTER_CUBIC)

    grayscale_resize_test_license_plate = cv2.cvtColor(
        resize_test_license_plate, cv2.COLOR_BGR2GRAY)

    gaussian_blur_license_plate = cv2.GaussianBlur(
        grayscale_resize_test_license_plate, (5, 5), 0)

    new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang ='eng',
    config ='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    final_str_plate = "".join(new_predicted_result_GWT2180.split()).replace(":", "").replace("-", "")
    return final_str_plate
    

