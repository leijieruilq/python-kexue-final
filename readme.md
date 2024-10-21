# Python科学计算大作业项目源码

## 依赖第三方库

1.tushare

2.torch

3.os

4 pandas

5 numpy

6 matplotlib

7 sys

8 yaml

9 warnings

10 datetime

## 使用方式

1) 获取股票信息:使用gupiao_get.py，保存为code.xlsx。

2) tcaffn.yaml:进行多元预测的初始设置:
   
>> choose_channels(选择要预测的多元时间列)，例如第[3,4,5]列（建议不要修改）:  
>>  - 3  
>>  - 4  
>>  - 5

>> read_path: code.xlsx

>> save_path: "model_save/model.pt" 

>> excel_path: "pred.xlsx"

>> current_timestamp_str: 当前的准确时间，例如'2024-10-21 13:15'。

>> device:训练模型的设备，如果不支持"cuda"环境，可以设置为"cpu"。

3) 运行tcaffn_predict.py:进行多元预测,并且记录训练和预测结果。