import os

import glob
import logging
import shutil

import numpy as np

from bing_image_downloader.downloader import download
from PIL import Image

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_images(path: str, list_actors: list, limit_load: int = 15) -> None:
    """
    Загрузка изображений для модели
    :param path: путь до датасета
    :param list_actors: словарь с именами актеров/актрис
    :param limit_load: ограничение на кол-во скачиваемых изображений
    :return: None
    """
    logging.info('Clean the folder')
    # удаляем папку с датасетом
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

    logging.info('Download images of actors by Bing')
    # пройдемся по каждому имени
    for face in list_actors:
        str_face = f'face {face}'
        # выгрузим 15 фотографий по текущему имени
        download(str_face,
                 limit=limit_load,
                 output_dir=path,
                 adult_filter_off=True,
                 force_replace=False,
                 timeout=60,
                 verbose=False)
        os.rename(path + '/' + str_face, path + '/' + face)
    logging.info('Completing the loading of actor images')


def resize_images(image: Image, size_new: int) -> np.array:
    """
    Изменение размера изображения
    :rtype: object
    :param image: изображение
    :param size_new: размер изображения по одной из сторон
    :return: изображение
    """
    size = image.size
    coef = size_new / size[0]
    # изменяем размер изображения
    resized_image = image.resize(
        (int(size[0] * coef), int(size[1] * coef)))
    resized_image = resized_image.convert('RGB')
    return resized_image


def format_images(path: str, list_actors: list, size_new: int) -> None:
    """
    Форматирование размера изображений
    :param size_new: размер изображения по одной из сторон
    :param path: путь до папки с датасетом
    :param list_actors: словарь с именами актеров/актрис
    :return: None
    """
    logging.info('Formatting the image of actors')
    # пройдемся по каждому имени
    for face in list_actors:
        # выгрузим все название файлов из папки
        files = glob.glob(f'{path}/{face}/*')
        for file in files:
            try:
                file_img = Image.open(file)
                resized_image = resize_images(file_img, size_new)
                resized_image.save(file)
            except Exception as ex:
                logging.info(f'Remove image {file} of {face}\nmessage: {ex}')
                os.remove(file)
