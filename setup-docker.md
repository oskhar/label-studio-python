Pertama build dulu image

```bash
docker build -t my-tensorflow-image .
```

Jalankan docker untuk program apapun yang ingin dijalankan
```bash
docker run -it --rm my-tensorflow-image python3 run-cnn.py
```
