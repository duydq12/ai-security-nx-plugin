FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gdb \
    gdbserver \
    rsync \
    openssh-server \
    git \
    cmake \
    g++-12 \
    unzip \
    python3.10 python3-pip wget \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm /usr/bin/g++ && rm /usr/bin/gcc && ln -s /usr/bin/g++-12 /usr/bin/g++ && ln -s /usr/bin/gcc-12 /usr/bin/gcc

RUN  wget https://updates.networkoptix.com/metavms/39242/sdk/metavms-metadata_sdk-5.1.5.39242-universal.zip && \
    unzip metavms-metadata_sdk-5.1.5.39242-universal.zip -d /opt
    
# Create a user for development (optional but recommended)
RUN useradd -m developer -s /bin/bash && \
    echo "developer:password" | chpasswd && \
    adduser developer sudo

RUN apt-get update &&  apt install sudo

# Configure SSH server
RUN mkdir /var/run/sshd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
# Or for the 'developer' user:
# RUN echo 'PermitUserLogin developer' >> /etc/ssh/sshd_config
# RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config # Ensure this is enabled if using password auth

# Expose SSH port
EXPOSE 22 7777

# Set a default command (e.g., start SSH server)
CMD ["/usr/sbin/sshd", "-D"]