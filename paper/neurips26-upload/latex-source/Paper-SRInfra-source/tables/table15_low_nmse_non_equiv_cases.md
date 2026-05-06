# Low-NMSE but non-equivalent symbolic-regression cases.

| Algorithm   | Dataset   |   Seed |   ID NMSE |   OOD NMSE | Ground truth              | Predicted expression   |   SYM-F |
|:------------|:----------|-------:|----------:|-----------:|:--------------------------|:-----------------------|--------:|
| gplearn     | Nguyen-9  |      2 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| dso         | Nguyen-9  |      4 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| pysr        | Nguyen-9  |      0 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| dso         | Nguyen-9  |      1 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| gplearn     | Nguyen-9  |      1 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| dso         | Nguyen-9  |      2 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| dso         | Nguyen-9  |      3 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
| dso         | Nguyen-9  |      0 |         0 |          0 | sin(x0) + sin(pow(x1, 2)) | sin(x0) + sin(x0**2)   |  0.4238 |
