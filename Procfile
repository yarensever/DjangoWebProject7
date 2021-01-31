web: gunicorn --workers=3 --threads=5 --worker-class=gthread DjangoWebProject7.wsgi --preload
worker: heroku run python views.py
