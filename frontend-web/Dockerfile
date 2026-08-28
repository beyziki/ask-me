# Geliştirme amaçlı Dockerfile: Vite dev server'ı container içinde çalıştırır,
# docker-compose.yml içindeki bind mount sayesinde kaynak kod değişiklikleri
# anında yansır (hot reload). Üretim/deploy için ayrı bir çok-aşamalı
# (multi-stage, nginx tabanlı) Dockerfile daha uygun olur, ama bu proje
# offline/lokal bir okul projesi olduğu için buna gerek yok.
FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
