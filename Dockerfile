FROM jekyll/jekyll:latest                                            

RUN apk add nginx

RUN gem sources --remove https://rubygems.org/
RUN gem source -a https://gems.ruby-china.com/
RUN mkdir /svr/jekyll -p
COPY . /svr/jekyll/
COPY nginx.conf /etc/nginx/
RUN chown jekyll:jekyll /svr/jekyll -R


WORKDIR /svr/jekyll
RUN cd /svr/jekyll
RUN ls -l
RUN bundle add webrick

#RUN adduser -D -g 'www' www
#RUN mkdir /www
#RUN chown -R www:www /var/lib/nginx
#RUN chown -R www:www /www

EXPOSE 4000
#CMD [ "/bin/bash" ]