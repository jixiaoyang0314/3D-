# Official Class Mapping

Date: 2026-07-08

This document records how the generated YOLO class names were mapped to the
official competition object IDs shown in the supplied object list images.

## Applied Mapping

| YOLO class index | Old generated name | Official ID | Object |
| --- | --- | --- | --- |
| 0 | banana | CD003 | banana |
| 1 | bb_0 | CB004 | potato chips |
| 2 | bb_1 | CC004 | bottled water |
| 3 | bb_2 | CC003 | boxed milk |
| 4 | bb_3 | CB003 | ham sausage |
| 5 | cup | CA003 | cup |
| 6 | new_0 | CB002 | melon seeds |
| 7 | new_1 | CB003 | ham sausage |
| 8 | new_2 | CC003 | boxed milk |
| 9 | part_0 | CA002 | earphones |
| 10 | part_2 | CC004 | bottled water |
| 11 | part_3 | CB004 | potato chips |
| 12 | part_1 | CC002 | bottled drink |
| 13 | part_4 | CB002 | melon seeds |
| 14 | part_5 | CB003 | ham sausage |
| 15 | part_6 | CC001 | canned drink |
| 16 | part_7 | CC003 | boxed milk |
| 17 | pinzi | CC004 | bottled drinking water |
| 18 | shupian+huotuichang_1 | CB004 | potato chips |
| 19 | shupian+huotuichang_0 | CB003 | ham sausage |
| 20 | yijia | CA004 | clothes hanger |

## Merged Official Dataset

Use this dataset for official-ID training:

```text
D:\机器人识别大赛\yolo_dataset_official\data.yaml
```

This dataset rewrites label class indices into 11 merged official classes:

| New class index | Official ID |
| --- | --- |
| 0 | CA002 |
| 1 | CA003 |
| 2 | CA004 |
| 3 | CB002 |
| 4 | CB003 |
| 5 | CB004 |
| 6 | CC001 |
| 7 | CC002 |
| 8 | CC003 |
| 9 | CC004 |
| 10 | CD003 |

For example, all old melon-seed labels from `new_0` and `part_4` now become
class index `3`, whose name is `CB002`.

All old bottled-drinking-water labels from `bb_1`, `part_2`, and `pinzi`
now become class index `9`, whose name is `CC004`.

## Official Classes Not Present In The Current Labeled Dataset

These official object IDs were visible in the supplied object list but do not
currently have a confidently matched labeled class in `yolo_dataset/data.yaml`:

| Official ID | Object |
| --- | --- |
| CA001 | brush |
| CB001 | chocolate |
| CD001 | peach |
| CD002 | apple |
| CD004 | pear |
