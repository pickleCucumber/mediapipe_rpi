# mediapipe_rpi
Последовательность команд и переписание библиотечного detect.py файла для распознавания и совместимой работы с raspberry pi 4 + камера 2 версии
Берем чистую microSD и записываем образ Raspberry PI OS Bookworm(64-bit)
Далее в командной строке:
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
python3 -m pip install -r requirements.txt --break-system-packages
pip install mediapipe --break-system-packages
pip install opencv-python --break-system-packages
sudo python -m pip install --upgrade pip --break-system-packages
sudo apt update
sudo apt install libatlas3-base
sudo apt install libcap-dev
git clone https://github.com/googlesamples/mediapipe.git
# Это если хотите проверить работу распознавания объектов
cd mediapipe/examples/object_detection/raspberry_pi
wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
python detect.py
# Далее распознавание жестов рук
cd /home/rpi/mediapipe/examples/hand_landmarker/raspberry_pi
sh setup.sh
# Далее запускаем test.py
