FROM jekyll/jekyll:latest                                            

RUN apk add nginx

RUN gem sources --remove https://rubygems.org/
RUN gem source -a https://gems.ruby-china.com/
#RUN gem sources -a http://mirrors.aliyun.com/rubygems/
RUN mkdir /svr/jekyll -p
COPY . /svr/jekyll/
COPY nginx.conf /etc/nginx/
RUN chown jekyll:jekyll /svr/jekyll -R

COPY entrypoints.sh /
RUN chmod +x /entrypoints.sh

WORKDIR /svr/jekyll

EXPOSE 4000
#CMD [ "/bin/bash" ]