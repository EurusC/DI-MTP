# DI-MTP

## 数据预处理
我们提供了mask的CSV和Pickle文件，出于商业隐私考虑，我们不能直接提供完整的数据集。为了更好地展现流程，我们截取了一部分数据，并对其 mmsi、lon、lat、timestamp 等关键信息做了处理，您可以将自己的数据处理成类似的形式。

### 1. 生成 Pickle 文件

为将原始AIS数据转换为可供模型使用的格式，首先需要将其转换为Pickle文件。具体步骤如下：

- 将需要的CSV文件放入 `orig_data` 文件夹中，按照场景分类（如 `orig_data/zhoushan/2022-10-1.csv`）。该文件夹可以包含多个CSV文件。
  
- 运行 `data_preprocess.py` 脚本来生成Pickle文件。脚本中需要修改经纬度范围，以确保数据处理的区域正确。特别是在 `load_csv` 函数中调整以下参数：

  ```python
  lon_min = [经度最小值]
  lon_max = [经度最大值]
  lat_min = [纬度最小值]
  lat_max = [纬度最大值]

### 2. 模型训练
要训练自己的模型，请使用 run.py 脚本，以下是主要参数的说明：

- obs_length：观察轨迹长度，指每次输入数据的时长。

- pred_length：预测轨迹长度，指模型需要预测的时间段。

- epoch：训练的迭代次数。

- scene：选择需要训练的场景（例如 "zhoushan"）。

- cuda：使用哪个GPU加速训练（如 0）。

#### 训练前的准备
- 在 run.py 中的 main 方法中，您需要根据您的数据集进行如下修改：
    - 设置 lon_min、lon_max、lat_min 和 lat_max 来确保训练数据的地理位置范围符合您的需求。
    - 修改 batches_path，指向包含数据的路径。

#### 训练
```shell
python run.py --obs_len 6 --pred_len 12 --scene 'mask' --cuda 0
  