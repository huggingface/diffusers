FROM mlp-docker-snapshot.artifactory.michelin.com/acpt-pytorch-2.0-cuda11.7:latest
ENV NO_PROXY='artifactory.michelin.com,gitlab.michelin.com'
ENV HTTP_PROXY="http://proxy-weu.aze.michelin.com:80"
ENV HTTPS_PROXY="http://proxy-weu.aze.michelin.com:80"
ENV http_proxy="http://proxy-weu.aze.michelin.com:80"
ENV https_proxy="http://proxy-weu.aze.michelin.com:80"

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV http_proxy=""
ENV https_proxy=""