
# DSSM

双塔结构，通过特征配置文件可以指定特征放在哪个特征塔，如下:

```
# 用户 ID 放在左塔
transform.add_categorical_identity('u_omgid', [])
transform.add_embedding('u_omgid.identity', ['dssm1'], 200)

# 物品 ID 放在右塔
transform.add_embedding('i_rowkey.identity', ['dssm2'], 100)
```

最后一层计算相似度有多种计算方法： dot, concat, cosine. 通过下面的方法配置：
```
model_type='dssm'
dssm_mode='dot'
```
