FROM python:3.12.1-alpine

WORKDIR /usr/src/app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
