FROM ubuntu:18.04

RUN apt-get update && apt-get install -y debconf-utils \
 && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 gcc cmake \
    dialog apt-utils software-properties-common
#RUN add-apt-repository -y ppa:keithw/mahimahi 

RUN sysctl -w net.ipv4.ip_forward=1 \
    && apt-get install -y mahimahi apache2 wget


RUN wget 'https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz' \
    && apt-get install -y python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip \
    && tar xvzf selenium-2.39.0.tar.gz


RUN cd selenium-2.39.0 \
    && python setup.py install \
    && echo 'DBUS_SESSION_BUS_ADDRESS=/dev/null' > /etc/init.d/selenium

RUN apt-get install -y fonts-liberation libasound2 libnspr4 libnss3 xdg-utils

RUN pip install pyvirtualdisplay==0.2 \
    && wget 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb' \
    && dpkg -i google-chrome-stable_current_amd64.deb 
#RUN apt-get -f -y install

RUN apt-get -y install python-pip python-dev 
    
#RUN pip install tensorflow tflearn

RUN apt-get -y install python-h5py python-scipy python-matplotlib

RUN git clone https://github.com/hongzimao/pensieve.git

RUN cd pensieve \
    && cp video_server/myindex_*.html /var/www/html \
    && cp video_server/dash.all.min.js /var/www/html \ 
    && cp -r video_server/video* /var/www/html \ 
    && cp video_server/Manifest.mpd /var/www/html \
    && mkdir cooked_traces \
    && mkdir run_exp/results \
    && mkdir real_exp/results
