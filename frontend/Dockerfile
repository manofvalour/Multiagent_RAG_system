# Frontend static file server using Nginx
FROM nginx:alpine

# Remove default nginx config
RUN rm -rf /etc/nginx/conf.d/*

# Copy nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy static frontend file
COPY frontend/rag-frontend.html /usr/share/nginx/html/index.html

# Create assets directory for future use
RUN mkdir -p /usr/share/nginx/html/assets

# Health check — nginx listens on 80 inside the container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost || exit 1

EXPOSE 80