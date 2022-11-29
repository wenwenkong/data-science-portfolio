FROM python:3.7

WORKDIR /app

RUN pip install pandas scikit-learn xgboost flask gunicorn

ADD ./model ./model
ADD server.py server.py
ADD ./templates ./templates

EXPOSE 60000

CMD [ "gunicorn", "--bind", "0.0.0.0:60000", "server:app" ]
