# Build context is the REPO ROOT (see docker-compose.yml), so the root
# .dockerignore (frontend/node_modules, frontend/dist, ...) applies here too.
FROM node:20-alpine

WORKDIR /app

# Installed before the rest of the source so this layer only rebuilds when a
# dependency actually changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .

EXPOSE 5173
CMD ["npm", "run", "dev"]
