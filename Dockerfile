# Utiliser une image Python officielle
FROM python:3.12-slim

# Créer un utilisateur non-root
RUN useradd -m -u 1000 appuser

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Créer et activer l'environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installer les dépendances dans l'environnement virtuel
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Changer les permissions
RUN chown -R appuser:appuser /app

# Passer à l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["python", "main.py"]
