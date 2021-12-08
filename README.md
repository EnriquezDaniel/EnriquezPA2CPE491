# Enriquez PA2 CPE491 / CS442 Cloud Computing



How to build:
> cd EnriquezPA2CPE491
> sudo docker build -t enriquezpa2docker .
> sudo docker tag enriquezpa2docker enriquezpa2docker/pa2
> sudo docker login
> sudo docker push enriquezpa2docker/pa2

How to pull from docker:
> sudo docker login
> 'Enter credentials'
> sudo docker pull enriquezdaniel/pa2

How to run image:
> sudo docker run enriquezdaniel/pa2
