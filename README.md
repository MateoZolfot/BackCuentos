docker compose up           # corre en :8000
curl -F "file=@cat.jpg" -F "prompt=..." http://localhost:8000/generate
