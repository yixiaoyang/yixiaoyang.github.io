#!/bin/bash
cd /svr/jekyll
bundle add webrick
jekyll build
nginx -g 'daemon off;'