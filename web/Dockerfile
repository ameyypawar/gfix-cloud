# web/Dockerfile
# Next.js standalone build. Build context: project root (.).
# NEXT_PUBLIC_API_URL is inlined at build time (default: http://localhost:8080
# so the browser on the host reaches the API through the mapped port).

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: deps — install node_modules
# ────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS deps
WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: builder — compile Next.js
# ────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY web/ .

ARG NEXT_PUBLIC_API_URL=http://localhost:8080
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ────────────────────────────────────────────────────────────────────────────
# Stage 3: runner — minimal standalone image
# ────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Standalone output (includes a trimmed node_modules and server.js)
COPY --from=builder /app/.next/standalone ./
# Static assets (not included in standalone)
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
# Public assets
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000

CMD ["node", "server.js"]
