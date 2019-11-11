# EasyCTR
封装主流 CTR 模型，配置化的训练方式，统一的 serving 接口。

# 特色
- 简单易用，用户只需要提供几个配置文件即可使用，包括特征处理、模型选择、超参调节等等
- 模型众多，几乎包含所有主流 CTR 模型，简单易用，支持灵活组装各种模型
- 端到端的 Ensemble 模型训练
- 线上、线下特征一致，这是因为特征处理逻辑在模型中完成
- 训练数据为原始特征格式，特征处理逻辑在模型中完成，用户通过配置文件进行配置即可
- 导出模型的 serving 接口统一
- 基于 Tensorflow Estimator 接口实现
- 支持多 GPU 训练
- 支持 spark-fuel 分布式训练框架

# 支持的 CTR 模型
- LR
- DNN
- [Wide&Deep](https://arxiv.org/abs/1606.07792)
- [DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)
- [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- [FwFM](https://arxiv.org/abs/1806.03514)
- [AFM](https://arxiv.org/abs/1708.04617)
- [IAFM](https://arxiv.org/abs/1902.09757)
- [KFM](https://arxiv.org/abs/1807.00311)
- [NIFM](https://arxiv.org/abs/1807.00311)
- [NFM](https://arxiv.org/abs/1708.05027)
- [CCPM](https://dl.acm.org/citation.cfm?id=2806603)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [xDeepFM](https://arxiv.org/abs/1803.05170)
- [DCN](https://arxiv.org/abs/1708.05123)
- [IPNN](https://arxiv.org/abs/1807.00311)
- [KPNN](https://arxiv.org/abs/1807.00311)
- [PIN](https://arxiv.org/abs/1807.00311)
- [FGCNN](https://arxiv.org/abs/1904.04447)
- [FiBiNET](https://arxiv.org/abs/1905.09433)
- [AutoInt](https://arxiv.org/abs/1810.11921)
- [DeepX (model slots 模式)](https://github.com/xuzhezhaozhao/EasyCTR/tree/master/models/deepx)

具体用法参考[模型文档](https://github.com/xuzhezhaozhao/EasyCTR/tree/master/models/)

# 待实现的模型
- [FFM](https://arxiv.org/abs/1701.04099)
- [NFFM](https://arxiv.org/abs/1904.12579)
- [BST](https://arxiv.org/abs/1905.06874)
- [DIEN](https://arxiv.org/abs/1809.03672)
- [DSIN](https://arxiv.org/abs/1905.06482)

# 用法

## 输入数据格式
输入数据包含 3 种:
 1. 特征描述文件;
 2. 训练样本;
 3. 特征配置文件;


### 特征描述文件
TSV 格式（tab或空格分隔符），描述特征信息。

格式为：
```
特征组名 特征名 特征类型 特征组序号 特征序号
```

示例：
```
user.id u_omgid string 0 0
user.basic u_devtype string 1 0
user.basic u_sex string 1 1
user.basic u_age int 1 2
user.basic u_city string 1 3
user.basic u_province string 1 4
user.basic u_city_level string 1 5
user.basic u_data_date float 1 6
user.behavior u_history string_list 2 0
user.portrait u_h_top_tags string_list 3 0
user.portrait u_h_top_tags_valid_ratio float_list 3 1
user.portrait u_h_top_cat1 string_list 3 2
user.portrait u_h_top_cat1_valid_ratio float_list 3 3
user.portrait u_h_top_cat2 string_list 3 4
user.portrait u_h_top_cat2_valid_ratio float_list 3 5
user.portrait u_h_top_cat3 string_list 3 6
user.portrait u_h_top_cat3_valid_ratio float_list 3 7
user.portrait u_h_neg_tags string_list 3 8
user.portrait u_h_neg_tags_valid_ratio float_list 3 9
user.portrait u_h_neg_cat1 string_list 3 10
user.portrait u_h_neg_cat1_valid_ratio float_list 3 11
user.portrait u_h_neg_cat2 string_list 3 12
user.portrait u_h_neg_cat2_valid_ratio float_list 3 13
user.portrait u_h_neg_cat3 string_list 3 14
user.portrait u_h_neg_cat3_valid_ratio float_list 3 15
item.id i_rowkey string 0 0
item.basic i_kb_duration int 1 0
item.basic i_kb_media_v_level int 1 1
item.basic i_kb_score_classic_id string 1 2
item.basic i_kb_score_quality_id string 1 3
item.basic i_kb_pic_resolution string 1 4
item.basic i_kb_black_rate string 1 5
item.basic i_kb_resolution string 1 6
item.basic i_kb_pic_scale string 1 7
item.basic i_kb_first_recommend string 1 8
item.basic i_kb_sec_recommend string 1 9
item.basic i_kb_aspect float 1 10
item.basic i_kb_flag_minivideo int 1 11
item.basic i_kb_fresh_score int 1 12
item.basic i_kb_tone int 1 13
item.basic i_kb_pub_time int 1 14
item.basic i_kb_src_type string 1 15
item.basic i_kb_locinfo string_list 1 16
item.basic i_kb_title_scale string 1 17
item.basic i_kb_kb_status int 1 18
item.basic i_kb_is_kb_recommoned int 1 19
item.basic i_kb_is_kb_not_recommoned int 1 20
item.basic i_kb_tags string_list 1 21
item.basic i_kb_content_level string 1 22
item.basic i_kb_minivideo_src string 1 23
item.basic i_kb_cms_state string 1 24
item.basic i_kb_media_id string 1 25
item.basic i_kb_title_len int 1 26
item.basic i_kb_release_date string 1 27
item.basic i_kb_offline_date string 1 28
item.basic i_kb_cover_border_type string 1 29
item.basic i_kb_video_border_type string 1 30
item.basic i_kb_tv_episode string 1 31
item.basic i_kb_src string 1 32
item.basic i_kb_showtype_big_pic string 1 33
item.basic i_kb_is_org int 1 34
item.basic i_kb_media_category string 1 35
item.basic i_kb_media_category2 string 1 36
item.basic i_kb_safe_level string 1 37
item.basic i_kb_st_src string 1 38
item.basic i_kb_crawl_src string 1 39
item.stats_24h i_total_click_cnt_24h float 2 0
item.stats_24h i_total_exposure_cnt_24h float 2 1
item.stats_24h i_total_play_cnt_24h float 2 2
item.stats_24h i_total_valid_play_cnt_24h float 2 3
item.stats_24h i_total_play_duration_24h float 2 4
item.stats_24h i_total_valid_play_duration_24h float 2 5
item.stats_24h i_video_duration_24h float 2 6
item.stats_24h i_play_duration_ratio_24h float 2 7
item.stats_24h i_average_play_time_24h float 2 8
item.stats_24h i_ctr_24h float 2 9
item.stats_24h i_play_cnt_ratio_24h float 2 10
item.stats_7d i_total_click_cnt_7d float 3 0
item.stats_7d i_total_exposure_cnt_7d float 3 1
item.stats_7d i_total_play_cnt_7d float 3 2
item.stats_7d i_total_valid_play_cnt_7d float 3 3
item.stats_7d i_total_play_duration_7d float 3 4
item.stats_7d i_total_valid_play_duration_7d float 3 5
item.stats_7d i_video_duration_7d float 3 6
item.stats_7d i_play_duration_ratio_7d float 3 7
item.stats_7d i_average_play_time_7d float 3 8
item.stats_7d i_ctr_7d float 3 9
item.stats_7d i_play_cnt_ratio_7d float 3 10
item.stats_30d i_total_click_cnt_30d float 4 0
item.stats_30d i_total_exposure_cnt_30d float 4 1
item.stats_30d i_total_play_cnt_30d float 4 2
item.stats_30d i_total_valid_play_cnt_30d float 4 3
item.stats_30d i_total_play_duration_30d float 4 4
item.stats_30d i_total_valid_play_duration_30d float 4 5
item.stats_30d i_video_duration_30d float 4 6
item.stats_30d i_play_duration_ratio_30d float 4 7
item.stats_30d i_average_play_time_30d float 4 8
item.stats_30d i_ctr_30d float 4 9
item.stats_30d i_play_cnt_ratio_30d float 4 10
ctx.basic c_first_item_id string 0 0
ctx.basic c_algo_id string 0 1
ctx.basic c_channel string 0 2
ctx.basic c_current_week int 0 3
ctx.basic c_current_hour int 0 4
ctx.basic c_is_weekend int 0 5
ctx.basic c_video_position int 0 6
ctx.basic c_scene string 0 7
extra.basic e_play_duration float 0 0
extra.basic e_video_duration float 0 1
extra.basic e_scene string 0 2
extra.engage e_like int 1 0
extra.engage e_collect int 1 1
extra.engage e_share int 1 2
extra.engage e_enter_comment int 1 3
extra.engage e_comment int 1 4
extra.engage e_dislike int 1 5
```


### 训练样本

格式：
```
label|weight|用户特征|物品特征|上下文特征|额外信息
```

label：浮点数，表示样本标签
weight：浮点数，表示样本权重

**用户特征**/**物品特征**/**上下文特征** 格式一致，如下描述（注意：分隔符是 $）：
```
<特征组1>$<特征组2>$<特征组3>$...
```

**特征组**格式如下（注意：分隔符是 tab）：
```
<特征1> <特征2> <特征3> ...
```

## 配置文件
配置文件使用 python 脚本生成，使用时将 `tools/conf_generator/conf_generator.py` 文件拷贝到工作目录. 参考 data/conf.py 文件.

### 特征选择
先创建特征选择类：

```
assembler = Assembler()
```

通过 `Assembler` 的 `add_*` 方法添加特征, 共有 6 种特征类型:
 1. int
 2. float
 3. string
 4. string_list 变长，逗号分隔
 5. float_list  变长，逗号分隔
 6. weighted_string_list  变长，逗号分隔

#### 1. int
```
def add_int(self, iname, default_value)
```

#### 2. float
```
def add_float(self, iname, default_value)
```

#### 3. string
```
def add_string(self, iname, dict_file, min_count, oov_buckets=None)

# min_count 表示字符串的最小出现次数，小于这个阈值的丢弃或者 hash (当 oov_buckets > 0 时)
# oov_buckets > 0 时，小于 min_count 的字符串 hash 进大小为 oov_buckets 的桶内;
# top_k 为最大词典数, 默认不限制
```

#### 4. string_list
例如 tag_list 这种变长特征，逗号分隔
```
def add_string_list(self, iname, dict_file, min_count, width, top_k=None)

# width 表示长度, 不足则用 -1, top_k 为最大词典数, 默认不限制
```

#### 5. float_list 类型特征
```
def add_float_list(self, iname, default_value, width)

# width 表示长度, 不足则用 default_value
```


#### 6. weighted_string_list 类型特征
带权重的 string list 特征，例如画像特征，tag 会带有权重
```
def add_weighted_string_list(self, iname, dict_file, min_count, width, top_k=None, min_weight=0.0)

# width 表示长度，不足默认 -1，top_k 表示最大词典数，min_weight 为最小权重值
```

### 特征转换
先创建特征转换类：

```
transform = Transform()
```

共有 10 种特征转换方式：
 1. numeric: 连续特征处理，支持 log 转换, 归一化;
 2. bucketized: 连续特征分桶;
 3. embedding: 离散特征 embedding;
 4. shared_embedding: 多个特征共享 embedding
 5. pretrained_embedding: 加载预训练 embedding;
 6. indicator: one-hot or multi-hot;
 7. categorical_identity: int 型特征离散处理;
 8. cross: 特征交叉;
 9. weighted_categorical: 带权重的离散特征处理;
 10. attention: attention 结构

通过对应的 `Transform.add_*` 方法添加特征转换.

#### 1. numeric
TODO - 支持更多归一化方法

```
def add_numeric(self, iname, places=[],
                normalizer_fn=None, substract=0, denominator=1.0,
                oname=None)

# normalizer_fn 可以为 'log-norm', 'norm', 表示归一化方式, substract, denominator 表示归一化参数;
# normalizer_fn 为 'log-norm' 时, 转换公式为 (log(x+1.0) - substract) / denominator;
# normalizer_fn 为 'norm' 时, 转换公式为 (x - substract) / denominator;
# 输入是原始 int/float 类型的特征名;
# 输出是 numeric_column;
# 可以放在 wide, deep, 不可以作为 cross_column 输入;
# oname 默认为 iname + '.numeric'
```

#### 2. bucketized

```
def add_bucketized(self, iname, places, boundaries, oname=None)

# boundaries 为数值数组, 代表分桶边界, Buckets include the left boundary, and
# exclude the right boundary. Namely, boundaries=[0., 1., 2.] generates buckets
# (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
# 输入可以是原始 int/float 类型的特征名, 也可以是 numeric_column
# 返回的是 bucketized_column
# 可以放在 wide, deep, 也可以作为 cross_column 的输入

# oname 默认为 iname + '.bucketized'
```

#### 3. embedding

```
def add_embedding(self, iname, places, dimension, oname=None)

# dimension 为 embedding 维度
# 可以放在 deep
# 输入可以是 categorical_*_column, 也可以是原始 string/string_list 类型, string_list类型时使用 'mean' combiner
# oname 默认为 iname + 'embedding'
```

#### 4. shared_embedding
```
def add_shared_embedding(self, inames, places, dimension, oname=None)

# 多种特征共享 embedding;
# inames 为数组
# oname 默认为 '#'.join(inames) + '.embedding'
```

#### 5. pretrained_embedding

```
def add_pretrained_embedding(self, iname, places, dimension, trainable,
                             pretrained_embedding_file, oname=None)

# trainable 是否训练 embedding
# pretrained_embedding_file 预训练 embedding 文件, 格式为：
# 第一行包含头部: total, dimension
# 之后每行: item_id <num> <num> ...
# oname 默认为 iname + '.pretrained_embedding'
```

#### 6. indicator
```
def add_indicator(self, iname, places, oname=None)

# 可以直接用于 string_, string__list 类型的特征, int/float 类型的特征需要先用 identity 转换一下
transform.add_indicator('user.city, 'user.city.indicator', ['wide'])

# user.gender 是 int 类型的特征, 用 add_categorical_indentity 转换一下
transform.add_categorical_identity('user.gender', 'user.gender.identity', [], 3)
transform.add_indicator('user.gender.identity', 'user.gender.indicator', ['wide', 'deep'])

# oname 默认为 iname + '.indicator'
```

#### 7. categorical_identity
```
def add_categorical_identity(self, iname, places, num_buckets=-1, oname=None)

# num_buckets 代表离散化的类别数, string/string_list 可以用 -1, 代表使用词典大小, int/float 类型则必须指定大小
# 输入是原始 int/float/string/string_list 的特征
# oname 默认 iname + '.identity'
```

#### 8. cross
```
def add_cross(self, inames, hash_bucket_size, oname=None)

# 此时 inames 为特征名数组, 代表参与交叉的特征, hash_bucket_size 为 hash 空间大小
# oname 默认为 '#'.join(inames) + '.cross'
```


#### 9. weighted_categorical
```
def add_weighted_categorical(self, inames, places, oname=None)

# inames 包含两个元素，第 1 个为 categorical column 特征名, 第 2 个为权重列，为原始字符串
# oname 默认为 inames[0] + '.weighted'
```


#### 10. attention
```
def add_attention(self, inames, attention_query, dimension, attention_type='din',
                  attention_args=None, shared_embedding=False, oname=None)

# inames 中每个特征都会与attention_query特征做attention处理
#  attention_type
#    din: alibaba din 论文中的attention方式
#    mlp: 类似 din 方式，去除 weight scale 逻辑
#  attention_args: python dict 类型, attention 方式的自定义参数, 例如MLP层数
#  shared_embedding: 是否共享 embedding
#
#  Note: 会同时添加 attention 特征（如浏览历史）和 query 特征（如排序物品ID）
```

# 编译
运行环境：
- python 2.7.x
- tensorflow 1.14.0

因为需要编译[自定义 op](https://www.tensorflow.org/guide/extend/op), 所以可能需要[源码编译 tensorflow](https://www.tensorflow.org/install/install_sources)。

serving 时需要用到自定义 op，所以需要重新编译 tesorflow serving，将自定义 op 编译进最后的二进制中。

# Demo
编译 tensorflow op:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

执行 `oh_my_go.sh` 脚本：
    $ ./oh_my_go.sh


# Road Map
- 完善使用方法文档


# 感谢
- https://github.com/shenweichen/DeepCTR
- http://git.code.oa.com/mmrecommend/deepx_rank_tf.git
