# установим образ face_recognition
FROM animcogn/face_recognition:latest

# скопируем файл с необходимыми библиотекми, которые хотим установить
COPY ./requirements.txt /root/requirements.txt

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r /root/requirements.txt

# создание рабочей директории
WORKDIR /root/docker_test

# копирование всех файлов, которые не указаны в dockerignore в новую директорию
COPY . /root/docker_test

# запуск скрипта, для нового проекта поставить train.py и сделать key_load_img: True в файле params.yaml
CMD ["python", "train.py"]
