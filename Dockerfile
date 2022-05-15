FROM python:3.7

COPY ./service/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY ./service /app

ENTRYPOINT [ "python3" ]

CMD [ "service.py" ]