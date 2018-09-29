# Sets up dependencies for unrestricted-adversarial-examples
# then runs all tests.
FROM ubuntu:18.04

# Install keyboard-configuration separately to avoid travis hanging waiting for keyboard selection
RUN \
    apt -y update && \
    apt install -y keyboard-configuration && \

    apt install -y \
        python-setuptools \
        python-pip \
        python3-dev \
        wget \
        unzip && \

    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install tox && \


# pre-install dependencies
COPY . /usr/local/unrestricted-adversarial-examples/
RUN cd /usr/local/unrestricted-adversarial-examples && \
    tox --notest

WORKDIR /usr/local/unrestricted-adversarial-examples/
CMD ["tox"]