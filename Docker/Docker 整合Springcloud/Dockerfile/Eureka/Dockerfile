FROM docker.io/kdvolder/jdk8
VOLUME /tmp
ADD app.jar /app.jar
EXPOSE 7000
ENTRYPOINT ["java","-jar","/app.jar"]