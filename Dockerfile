# use pytorch image with specified version with all dependencies installed
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# specify wokring directory
WORKDIR /code
# VOLUME /src

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src /code/src

# replace opencv-python with headless version for pysolotools
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
