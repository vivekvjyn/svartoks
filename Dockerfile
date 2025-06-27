FROM       ubuntu:18.04
LABEL authors="Vivek Vijayan"
LABEL version="18.04"
LABEL description="Tokenizer for pitch time series"
RUN apt-get update
RUN apt-get install -y openssh-server nmap sudo telnet sssd
RUN mkdir /var/run/sshd
RUN echo 'root:xxxxxxxxxxxx' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 22
CMD    ["/usr/sbin/sshd", "-D"]

FROM       python:3.10.16 AS app
ADD requirements.txt .
RUN pip install -r requirements.txt
