FROM python:3.10

ADD Telebot.py .
ADD data.zip .
ADD Bookclass.zip .

RUN pip install requests
RUN pip install telebot
RUN pip install numpy
RUN pip install pandas
RUN pip install nltk
RUN pip install tensorflow
RUN pip install scikit-learn
RUN pip install seaborn

CMD ["python", "./Telebot.py"]