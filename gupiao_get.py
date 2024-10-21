# In[]
import tushare as ts
import pandas as pd
ts.set_token('f75468fcaced81a01f116789dceb55d9b427eccae2195adeae09e036')
code_name = '601058'
#   - '601058'
#   - '002935'
#   - '600418'
#   - '601009'
#   - '000001'
# 初始化pro接口
pro = ts.pro_api()
df = ts.get_k_data(code_name,#,start='2024-02-19',end='2024-02-23',
                   ktype="60")
print(df)
df.to_excel(code_name+'.xlsx')
# %%
