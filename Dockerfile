FROM python:3.10-slim
MAINTAINER Kuang Hangdong<khd401208163@gmail.com>

ADD ./ /FL
WORKDIR /FL

RUN pip install --upgrade pip
RUN pip install -r requirements.txt