import pickle
import re
import statistics as st
import zipfile

image_w = 2048
image_h = 700
image_path = "data/obj/"


# распаковка картинок
with zipfile.ZipFile("93.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)


# десереализация словарей
with open('93.a.cache.pkl', 'rb') as pkl_file:
    dic_a = pickle.load(pkl_file)
with open('93.b.cache.pkl', 'rb') as pkl_file:
    dic_b = pickle.load(pkl_file)


# создадим список из имен картинок, которые есть в папке
from os import listdir
from os.path import isfile, join
image_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]


# создадим txt-файл с перечислением имён 80% картинок для тренировочной выборки
image_names_train = image_names[:8000]
file = open("data/train.txt", "w")
for image in image_names_train:
    file.write("data/obj/%s\n" % (image))
file.close()


# создадим txt-файл с перечислением имён 10 картинок для теста
image_names_train = image_names[9990:]
file = open("data/test.txt", "w")
for image in image_names_train:
    file.write("data/obj/%s\n" % (image))
file.close() 


# Функция получения данных из словаря dic_a
def label_a(image_name):
# Переименуем координаты для удобства расчетов в формате xbr = x bottom right и т.д.
    xbl = dic_a[image_name][0]['coordinates'][0]['x']
    ybl = dic_a[image_name][0]['coordinates'][0]['y']
    xbr = dic_a[image_name][0]['coordinates'][1]['x']
    ybr = dic_a[image_name][0]['coordinates'][1]['y']
    xtl = dic_a[image_name][0]['coordinates'][3]['x']
    ytl = dic_a[image_name][0]['coordinates'][3]['y']
    xtr = dic_a[image_name][0]['coordinates'][2]['x']
    ytr = dic_a[image_name][0]['coordinates'][2]['y']
# посчитаем нужные для darknet значения
    x_center_a = st.mean([xbl,xbr,xtl,xtr])/image_w
    y_center_a = st.mean([ybl,ybr,ytl,ytr])/image_h
    width_a = (st.mean([xbr,xtr])-st.mean([xbl,xtl]))/image_w
    height_a = (st.mean([ytr,ytl])-st.mean([ybr,ybl]))/image_h
    return x_center_a, y_center_a, width_a, height_a


# Функция получения данных из словаря dic_b
def label_b(image_name):
# Достанем координаты номера при помощи регулярных выражений. Ищем числа между "xmin": и запятой,
# Берем первое совпадение
    xmin = int(re.compile(r'\"xmin\": (\d+),').findall(dic_b[image_name])[0])
    ymin = int(re.compile(r'\"ymin\": (\d+),').findall(dic_b[image_name])[0])
    xmax = int(re.compile(r'\"xmax\": (\d+),').findall(dic_b[image_name])[0])
    ymax = int(re.compile(r'\"ymax\": (\d+)}').findall(dic_b[image_name])[0])
# Теперь рассчитаем нужные для darknet значения, аналогично словарю dic_a
    x_center_b = st.mean([xmin,xmax])/image_w
    y_center_b = st.mean([ymin,ymax])/image_h
    width_b = (xmax-xmin)/image_w
    height_b = (ymax-ymin)/image_h
    return x_center_b, y_center_b, width_b, height_b

# Цикл, который для каждой картинки создаст текстовый файл с разметкой для обучения модели
# в формате <object-class> <x_center> <y_center> <width> <height>
for image_name in image_names:
    no_label_a = False
    no_label_b = False
    try:
        x_center_a, y_center_a, width_a, height_a = label_a(image_name)      
    except:
        no_label_a = True
        pass
    try:
        x_center_b, y_center_b, width_b, height_b = label_a(image_name)      
    except:
        no_label_a = True
        pass
# если не удалось достать информацию по картинке из словарей, выводим имя картинки, но таких случаев нет
# можно добавить обработку такого случая, например, удалить картинку, но я пока не стал этого делать
    if no_label_a == True and no_label_b == True:
        print(image)
# если удалось данные из обоих словарей, объединим результаты предсказаний, посчитав среднее значение
    if no_label_a == False and no_label_b == False:
        x_center = st.mean([x_center_a, x_center_b])
        y_center = st.mean([y_center_a, y_center_b])
        width = st.mean([width_a, width_b])
        height = st.mean([height_a, height_b])
# если данные только в первом словаре
    if no_label_a == False and no_label_b == True:
        x_center = x_center_a
        y_center = y_center_a
        width = width_a
        height = height_a      
# если данные только во втором словаре
    if no_label_a == True and no_label_b == False:
        x_center = x_center_b
        y_center = y_center_b
        width = width_b
        height = height_b
# переименуем image_name, удалив .jpg и добавив .txt
    image_name = image_name.split('.')[0]
    image_name += '.txt'
    txt_path = image_path+image_name
# запишем данные в файл
    file = open(txt_path, "w") 
    file.write("0 %s %s %s %s" % (str(x_center),str(y_center),str(width),str(height)))
    file.close() 