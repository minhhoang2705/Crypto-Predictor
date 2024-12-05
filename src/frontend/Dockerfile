# Build stage
FROM node:16-alpine as build

WORKDIR /app

# Copy package files
COPY src/frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/frontend/ ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy build files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80 443

# Start Nginx
CMD ["nginx", "-g", "daemon off;"] 