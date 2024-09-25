FROM python:3.12.1-slim

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python", "run_api.py" ]
