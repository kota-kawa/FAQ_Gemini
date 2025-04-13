FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

#CMD ["python", "app.py"]
########## デバッグ用の実行 ##############
# Flask の環境変数を設定（app.py がエントリーポイントの場合）
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--reload"]