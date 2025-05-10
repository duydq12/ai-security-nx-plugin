## Run clion-dev-container
```
docker compose up -d --build
```

## Modification contanier
```
docker compose exec -u developer clion-dev-container bash

sudo pip3 install "conan<2"
# pass: password

conan config init
conan profile update settings.compiler=gcc default
conan profile update settings.compiler.version=12 default
conan profile update settings.compiler.libcxx=libstdc++11 default
```

# Build evironment
docker run -it -v /home/ubuntu/H_project/2025/nx_open_integrations/conan:/home/developer/.conan --user developer  mx_meta_plugin bash 