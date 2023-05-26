import matplotlib.pyplot as plt 
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import telebot
# -*- coding: cp1251 -*-

bot = telebot.TeleBot('')

@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(message.chat.id, 'Пришли фотографию автомобиля с номером!')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo = message.photo[-1]  # Берем последнюю (наибольшего размера) фотографию
    file_id = photo.file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    # Загружаем фотографию
    downloaded_file = bot.download_file(file_path)

    # Сохраняем фотографию на компьютере
    save_path = 'Project3/cars/1.jpg'  # Укажите путь, куда сохранить фотографию
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, "Фотография сохранена!")
    carplate_img_rgb = cv2.imread('Project3/cars/1.jpg')
    carplate_img_rgb = cv2.cvtColor(carplate_img_rgb, cv2.COLOR_BGR2RGB)
    

    carplate_haar_cascade = cv2.CascadeClassifier('Project3/haar_cascades/haarcascade_russian_plate_number.xml')

    #Вырезаем нужную область
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in carplate_rects:
        carplate_img = carplate_img_rgb[y+10:y+h-10, x+15:x+w-15]

    carplate_extract_img = carplate_img

    width = int(carplate_extract_img.shape[1] * (150 / 78))
    height = int(carplate_extract_img.shape[0] * (150 / 78))
    dim = (width, height)
    resized_image = cv2.resize(carplate_extract_img, dim, interpolation = cv2.INTER_AREA)

    carplate_extract_img = resized_image
    plt.imsave("Project3/cars/res.jpg", carplate_extract_img)
    img = Image.open('Project3\\cars\\res.jpg')

    #Увеличиваем контрастность:
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.filter(ImageFilter.MedianFilter())
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    #Преобразуем в черно-белый рисунок:
    thresh = 130
    fn = lambda x : 255 if x > thresh else 0
    res = img.convert('L').point(fn, mode='1')
    res.save("Project3/cars/res.jpg")

    result = pytesseract.image_to_string(
        res,
        config='--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    print('Number:', result)

    bot.reply_to(message, f'Number: {result}')

        

bot.polling()