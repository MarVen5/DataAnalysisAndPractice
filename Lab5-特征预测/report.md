# **数据分析及实践 Lab 5**

马文宇 PB23061139

### **PART 1**

选择若干合适的机器学习算法进行训练，对特征 `TEACHBEHA` 进行预测，并汇总分析结果。

##### **S1**
数据信息和预处理：请概述所使用数据集的基本信息，并处理缺失值和异常值。

```python
import pandas as pd

# 加载数据
data = pd.read_csv('subdata.csv', index_col=0)

# 输出数据初始信息
print(f"1) Dataset shape: {data.shape}")  # 期望 (n, p)
print("\n2) Missing values per column:\n\n", data.isnull().sum()) # 缺失值
display(data.describe()) # 描述性
```
运行结果如下：
```
1) Dataset shape: (1089, 25)

2) Missing values per column:

 CNTSCHID              0
Region                0
STRATUM               0
LANGTEST             25
PRIVATESCH            0
SCHLTYPE             49
STRATIO             112
SCHSIZE             101
RATCMP1             119
RATCMP2              71
TOTAT                65
PROATCE             195
PROAT5AB            232
PROAT5AM            299
PROAT6              219
CLSIZE               41
CREACTIV             51
EDUSHORT             40
STAFFSHORT           45
STUBEHA              35
TEACHBEHA            35
SCMCEG               35
W_SCHGRNRABWT         0
W_FSTUWT_SCH_SUM      0
SENWT                 0
dtype: int64
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNTSCHID</th>
      <th>Region</th>
      <th>LANGTEST</th>
      <th>SCHLTYPE</th>
      <th>STRATIO</th>
      <th>SCHSIZE</th>
      <th>RATCMP1</th>
      <th>RATCMP2</th>
      <th>TOTAT</th>
      <th>PROATCE</th>
      <th>...</th>
      <th>CLSIZE</th>
      <th>CREACTIV</th>
      <th>EDUSHORT</th>
      <th>STAFFSHORT</th>
      <th>STUBEHA</th>
      <th>TEACHBEHA</th>
      <th>SCMCEG</th>
      <th>W_SCHGRNRABWT</th>
      <th>W_FSTUWT_SCH_SUM</th>
      <th>SENWT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.089000e+03</td>
      <td>1089.000000</td>
      <td>1064.000000</td>
      <td>1040.000000</td>
      <td>977.000000</td>
      <td>988.000000</td>
      <td>970.000000</td>
      <td>1018.000000</td>
      <td>1024.000000</td>
      <td>894.000000</td>
      <td>...</td>
      <td>1048.000000</td>
      <td>1038.000000</td>
      <td>1049.000000</td>
      <td>1044.000000</td>
      <td>1054.000000</td>
      <td>1054.000000</td>
      <td>1054.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.240055e+07</td>
      <td>72409.922865</td>
      <td>185.889098</td>
      <td>2.568269</td>
      <td>11.683976</td>
      <td>710.293522</td>
      <td>0.836264</td>
      <td>0.989038</td>
      <td>60.686035</td>
      <td>0.918203</td>
      <td>...</td>
      <td>27.866412</td>
      <td>1.129094</td>
      <td>0.114850</td>
      <td>0.292359</td>
      <td>-0.151848</td>
      <td>-0.099276</td>
      <td>0.295236</td>
      <td>6.322108</td>
      <td>392.693059</td>
      <td>4.591368</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.177777e+02</td>
      <td>5.023979</td>
      <td>93.179648</td>
      <td>0.622913</td>
      <td>4.728625</td>
      <td>443.718245</td>
      <td>0.948376</td>
      <td>0.072141</td>
      <td>32.285593</td>
      <td>0.243935</td>
      <td>...</td>
      <td>9.678761</td>
      <td>0.952438</td>
      <td>1.069071</td>
      <td>0.906954</td>
      <td>1.081774</td>
      <td>0.967054</td>
      <td>0.857401</td>
      <td>7.511681</td>
      <td>421.481247</td>
      <td>5.455284</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.240000e+07</td>
      <td>72401.000000</td>
      <td>156.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>-1.421200</td>
      <td>-1.455100</td>
      <td>-3.378500</td>
      <td>-2.090400</td>
      <td>-4.051800</td>
      <td>1.000000</td>
      <td>2.695650</td>
      <td>0.726240</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.240028e+07</td>
      <td>72406.000000</td>
      <td>156.000000</td>
      <td>2.000000</td>
      <td>8.690900</td>
      <td>384.500000</td>
      <td>0.440300</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>-0.688400</td>
      <td>-0.177600</td>
      <td>-0.789400</td>
      <td>-0.599275</td>
      <td>-0.186800</td>
      <td>1.933930</td>
      <td>138.468940</td>
      <td>1.404500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.240055e+07</td>
      <td>72411.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>11.176500</td>
      <td>648.000000</td>
      <td>0.708850</td>
      <td>1.000000</td>
      <td>57.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>28.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>0.349800</td>
      <td>-0.087100</td>
      <td>-0.117700</td>
      <td>0.904200</td>
      <td>3.550330</td>
      <td>226.947960</td>
      <td>2.578400</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.240082e+07</td>
      <td>72414.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>14.556500</td>
      <td>923.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>76.500000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>28.000000</td>
      <td>2.000000</td>
      <td>0.800600</td>
      <td>0.837100</td>
      <td>0.551500</td>
      <td>0.601500</td>
      <td>0.904200</td>
      <td>7.275120</td>
      <td>404.079840</td>
      <td>5.283480</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.240110e+07</td>
      <td>72419.000000</td>
      <td>608.000000</td>
      <td>3.000000</td>
      <td>51.578900</td>
      <td>2698.000000</td>
      <td>25.000000</td>
      <td>1.000000</td>
      <td>500.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>53.000000</td>
      <td>3.000000</td>
      <td>2.959500</td>
      <td>4.044200</td>
      <td>3.441000</td>
      <td>3.787900</td>
      <td>0.904200</td>
      <td>82.676870</td>
      <td>2052.424310</td>
      <td>60.043260</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
##### **Q2**

##### **S1.1** 缺失值填充

数值型特征 `(float64, int64)` → 中位数

分类型特征 `(object)` → 众数

```python
from sklearn.impute import SimpleImputer

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

data[num_cols] = num_imputer.fit_transform(data[num_cols])
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# 输出填充后的缺失值
print("3) Missing values after imputation:\n\n", data.isnull().sum())
```
运行结果如下：
```
3) Missing values after imputation:

 CNTSCHID            0
Region              0
STRATUM             0
LANGTEST            0
PRIVATESCH          0
SCHLTYPE            0
STRATIO             0
SCHSIZE             0
RATCMP1             0
RATCMP2             0
TOTAT               0
PROATCE             0
PROAT5AB            0
PROAT5AM            0
PROAT6              0
CLSIZE              0
CREACTIV            0
EDUSHORT            0
STAFFSHORT          0
STUBEHA             0
TEACHBEHA           0
SCMCEG              0
W_SCHGRNRABWT       0
W_FSTUWT_SCH_SUM    0
SENWT               0
dtype: int64
```

##### **S1.2**

箱线法：剔除落在 `[1%, 99%]` 范围之外的行

```python
lower, upper = 0.01, 0.99
print("4) The number of removed columns:\n")
for col in num_cols:
    q_low, q_high = data[col].quantile([lower, upper])
    before = data.shape[0]
    data = data[data[col].between(q_low, q_high)]
    after = data.shape[0]
    print(f"{col}: removed {before - after} rows")

# 输出新的数据维度
print(f"\n5) Shape after outlier removal: {data.shape}")
```
运行结果如下：
```
4) The number of removed columns:

CNTSCHID: removed 22 rows
Region: removed 8 rows
LANGTEST: removed 9 rows
SCHLTYPE: removed 0 rows
STRATIO: removed 22 rows
SCHSIZE: removed 22 rows
RATCMP1: removed 22 rows
RATCMP2: removed 10 rows
TOTAT: removed 19 rows
PROATCE: removed 0 rows
PROAT5AB: removed 0 rows
PROAT5AM: removed 0 rows
PROAT6: removed 10 rows
CLSIZE: removed 0 rows
CREACTIV: removed 0 rows
EDUSHORT: removed 0 rows
STAFFSHORT: removed 6 rows
STUBEHA: removed 10 rows
TEACHBEHA: removed 14 rows
SCMCEG: removed 10 rows
W_SCHGRNRABWT: removed 10 rows
W_FSTUWT_SCH_SUM: removed 18 rows
SENWT: removed 9 rows

5) Shape after outlier removal: (868, 25)
```

##### **S2**
数据集划分：按固定比例划分训练集 / 验证集 / 测试集，或者使用 k 折交叉验证法划分训练集 / 测试集（可调整随机种子以获得不同的划分方案）。

此处将采用训练集 70%，验证集 15%，测试集 15% 的划分方案。

```python
from sklearn.model_selection import train_test_split

X = data.drop(columns=['TEACHBEHA'])
y = data['TEACHBEHA']

# 划分训练集与临时集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# 将临时集分为：验证集与测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"1) Train: {X_train.shape[0]} samples\n")
print(f"2) Validation: {X_val.shape[0]} samples\n")
print(f"3) Test: {X_test.shape[0]} samples")
```
运行结果如下：
```
1) Train: 607 samples

2) Validation: 130 samples

3) Test: 131 samples
```
##### **S3**
机器学习算法模型：选择至少两种机器学习算法模型，作为主实验的比较方法，汇报它们的模型 / 算法信息，并在报告中引用相应的参考文献。

此处将选取 **线性回归** (LinearRegression) 与 **随机森林回归** (RandomForestRegressor)

参考文献：

**线性回归** (LinearRegression)：Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://arxiv.org/abs/1201.0490

**随机森林回归** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. https://link.springer.com/article/10.1023/A:1010933404324

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42)
}
print(models)
```
运行结果如下：
```
{'LinearRegression': LinearRegression(), 'RandomForest': RandomForestRegressor(random_state=42)}
```

##### **S4**
特征选择与处理：根据数据集本身和选用机器学习算法模型的特点，参考实验三的数据分析结论，选取合适的特征并进行一定适应性处理后作为模型输入。

对数值特征进行标准化 (StandardScaler)

构建 `scikit-learn Pipeline`

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 区分数值和分类型特征
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# 构建预处理器
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 构建 pipeline 函数
def make_pipeline(model):
    return Pipeline([
        ('preprocess', preprocessor),
        ('regressor', model)
    ])

# 对 LinearRegression 建立 pipeline 并预测前 3 条
pipe_test = make_pipeline(models['LinearRegression'])
pipe_test.fit(X_train, y_train)
print("Baseline predict sample:", pipe_test.predict(X_train.iloc[:3]))
```
运行结果如下：
```
Baseline predict sample: [ 0.30401184 -0.1026226   0.55354617]
```
##### **S5**

主实验：分别在选择的算法模型上进行训练（可直接使用机器学习库也可自行实现），确定评测指标（如 $MSE$ 等），汇报并分析它们在测试集上的结果。

此处在测试集上评估 $MSE$ 和 $R^2$

```python
from sklearn.metrics import mean_squared_error, r2_score

results = []
for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, r2))
    print(f"{name} - Test MSE: {mse:.4f}, R2: {r2:.4f}")
```
运行结果如下：
```
LinearRegression - Test MSE: 0.4713, R2: 0.2856
RandomForest - Test MSE: 0.4890, R2: 0.2588
```

##### **S6**

参数实验：对选用的模型，汇报并分析调整某些关键参数后所得到的实验结果。

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 5, 10]
}
pipe_rf = make_pipeline(models['RandomForest'])
search = GridSearchCV(pipe_rf, param_grid, cv=5, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
# 在测试集上验证最佳模型
best_rf = search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"Tuned RF - Test MSE: {mean_squared_error(y_test, y_pred_best):.4f}, R2: {r2_score(y_test, y_pred_best):.4f}")
```
运行结果如下：
```
Best params: {'regressor__max_depth': None, 'regressor__n_estimators': 200}
Tuned RF - Test MSE: 0.4801, R2: 0.2723
```
##### **S7**

结论分析：对使用方法的合理性和所得结论依据进行解释说明。

$\textbf{ANS}$

线性回归模型在测试集上的表现：  
$$ MSE = 0.4713 $$
$$ R^2 = 0.2856 $$

随机森林模型在测试集上的表现：  
$$ MSE = 0.4890 $$
$$ R^2 = 0.2588 $$

调参后随机森林模型在测试集上的表现：  
$$ MSE = 0.4801 $$
$$ R^2 = 0.2723 $$
通过改变 `n_estimators` 和 `max_depth`，$MSE$ 降低了 $1.82\%$，$R^2$ 提升了 $5.56\%$。

随机森林能够捕捉非线性关系，抗噪声能力较强，效果优于线性模型。  

线性回归模型训练速度快、易解释，但对数据分布和线性假设敏感。

### **PART 2**

近年来，大语言模型已经成为人工智能领域重要的生产力，其在数据分析领域同样展现出强大的能力，请利用大语言模型辅助完成以下任务。

##### **Q1**
登录 DeepSeek（ https://chat.deepseek.com ）或腾讯元宝（ https://yuanbao.tencent.com/chat ）对话窗口，设计合适的提示词，要求DeepSeek大语言模型基于数据集特征定义的元信息推测可能存在关联的特征，并比较不同设置下的输出结论：

##### **Q1.a** 关闭“深度思考（R1）”，并要求模型直接输出结论

输出如下：

---

以下是可能存在关联的特征组合推测：

##### 1. 学校特征之间的关联

- **SCHLTYPE (学校类型) 与 PRIVATESCH (私立学校)**  
  学校类型（1/2/3）可能与私立/公立属性有直接对应关系（例如类型2可能专指私立学校）。

- **SCHSIZE (学校规模) 与 STRATIO (学生教师比)**  
  规模较大的学校（如`SCHSIZE=1572`）可能学生教师比（`STRATIO`）更高（需验证是否存在正/负相关）。

- **SCHSIZE 与 CLSIZE (班级规模)**  
  学校规模可能影响平均班级人数（例如大规模学校可能分班更多，但班级人数未必显著增加）。

##### 2. 资源与教育质量关联

- **STRATIO (学生教师比) 与 TEACHBEHA (教师行为)**  
  更高的学生教师比可能导致教师行为压力（如负值的`TEACHBEHA`可能表示更多负面行为）。

- **STAFFSHORT (教职工短缺) 与 EDUSHORT (教育资源短缺)**  
  两者可能反映学校整体资源匮乏，存在正相关。

- **RATCMP1/RATCMP2 (资源分配指标) 与 PROATCE/PROAT5AB (学业表现)**  
  资源分配合理性可能影响学生学业成绩（如`RATCMP2=1.0`时`PROAT5AB`是否更高）。

##### 3. 学生行为与学校环境

- **STUBEHA (学生行为) 与 TEACHBEHA (教师行为)**  
  学生行为问题（如负值）可能与教师行为（如纪律管理）相互影响（可能存在负相关）。

- **SCMCEG (学校管理效率) 与 CREACTIV (学校创新活动)**  
  高效管理的学校可能更倾向于开展创新活动（如`SCMCEG`正值与`CREACTIV=1.0`的关系）。

##### 4. 区域与学校属性

- **Region (地区代码) 与 PRIVATESCH (私立学校)**  
  某些地区（如`72413`）可能私立学校更集中（需统计地区分布）。

- **STRATUM (分层变量) 与 SCHLTYPE (学校类型)**  
  分层可能基于学校类型或地区，需验证类别分布。

##### 5. 权重变量的潜在关联

- **W_SCHGRNRABWT (学校权重) 与 SCHSIZE (学校规模)**  
  大规模学校的样本权重可能更高（如`SCHSIZE=1572`对应`W_SCHGRNRABWT=16.53833`）。

##### 验证建议

1. **统计分析**  
   - 计算数值型特征的相关系数（如`SCHSIZE`与`STRATIO`）  
   - 对类别型特征进行卡方检验（如`PRIVATESCH`与`SCHLTYPE`）

2. **可视化方法**  
   - **散点图**：`SCHSIZE` vs `STRATIO`  
   - **箱线图**：`PRIVATESCH`分组下的`TEACHBEHA`分布  
   - **热力图**：资源指标（`RATCMP1`）与学业表现（`PROATCE`）的相关性

---

##### **Q1.b** 关闭“深度思考（R1）”，并要求模型逐步思考输出答案

输出如下：

---
##### 第一步：理解字段含义
通过观察字段名称和示例数据，初步判断字段类型和含义：
- **标识类**：CNTSCHID（学校ID）、Region（地区代码）、STRATUM（分层变量）
- **学校属性**：PRIVATESCH（公私类型）、SCHLTYPE（学校类型）、SCHSIZE（学校规模）、CLSIZE（班级规模）
- **资源指标**：STRATIO（师生比）、RATCMP1/RATCMP2（资源分配）、STAFFSHORT（教职工短缺）
- **教育质量**：PROATCE/PROAT5AB（学业表现）、TEACHBEHA（教师行为）、SCMCEG（管理效率）
- **权重类**：W_SCHGRNRABWT（学校权重）、W_FSTUWT_SCH_SUM（学生权重总和）

##### 第二步：识别数值型和类别型变量
| 类型       | 字段示例                  |
|------------|---------------------------|
| **数值型** | SCHSIZE, STRATIO, PROATCE |
| **类别型** | PRIVATESCH, SCHLTYPE      |

##### 第三步：分组关联推测

1. **学校基础属性组**
   - `SCHLTYPE`与`PRIVATESCH`  
     → 学校类型可能直接关联公私性质（如类型3=公立）
   - `SCHSIZE`与`STRATIO`  
     → 大规模学校可能师生比更高（需验证方向性）
   - `SCHSIZE`与`CLSIZE`  
     → 学校规模可能影响班级人数（非线性关系）

2. **资源与表现组**
   - `STRATIO`与`TEACHBEHA`  
     → 高师生比可能导致教师行为压力（负相关假设）
   - `RATCMP2`与`PROAT5AB`  
     → 资源分配合理性与学业成绩可能存在阈值效应（如RATCMP2=1.0时成绩跃升）

3. **行为与环境组**
   - `STUBEHA`与`TEACHBEHA`  
     → 学生行为与教师行为可能双向影响（需因果分析）
   - `SCMCEG`与`CREACTIV`  
     → 高效管理可能促进创新活动（正相关假设）

4. **区域特征组**
   - `Region`与`PRIVATESCH`  
     → 特定地区（如72413）私立学校比例更高（需频次统计）

##### 第四步：验证优先级排序
按业务重要性排序待验证关联：
1. **核心教育指标**  
   `STRATIO` → `PROATCE`（师生比对成绩的影响）
2. **资源分配有效性**  
   `RATCMP1` → `STAFFSHORT`（资源分配是否缓解短缺）
3. **行为传导机制**  
   `STUBEHA` → `SCMCEG` → `TEACHBEHA`（行为传导链条）

##### 第五步：输出关联矩阵建议
推荐检验的关联对：

| 特征1       | 特征2       | 关联类型       | 验证方法          |
|-------------|-------------|----------------|-------------------|
| SCHLTYPE    | PRIVATESCH  | 类别-类别      | 卡方检验          |
| STRATIO     | TEACHBEHA   | 数值-数值      | 皮尔逊相关系数    |
| SCHSIZE     | W_SCHGRNRABWT | 数值-数值    | 散点图+回归分析   |
| Region      | PRIVATESCH  | 类别-类别      | 频数分布表        |

##### 最终答案
基于逐步分析，**最可能存在的特征关联**包括：
1. **学校类型与性质**：`SCHLTYPE`和`PRIVATESCH`的强对应关系  
2. **规模效应**：`SCHSIZE`与`STRATIO`/`CLSIZE`的资源分配连锁反应  
3. **行为传导**：`STUBEHA`→`TEACHBEHA`→`PROATCE`的负向传导链  
4. **区域差异**：特定`Region`代码与私立学校聚集现象  

建议优先使用**分组统计+散点图矩阵**验证上述假设。

---

##### **Q1.c** 打开“深度思考（R1）”，无额外输出要求。

输出如下：

---
#### 一、关键特征分类
| 特征类型       | 包含字段示例                                |
|----------------|-------------------------------------------|
| **学校标识**   | CNTSCHID, Region, STRATUM                |
| **基础属性**   | PRIVATESCH, SCHLTYPE, SCHSIZE, CLSIZE    |
| **资源分配**   | STRATIO, RATCMP1, RATCMP2, STAFFSHORT    |
| **教育产出**   | PROATCE, PROAT5AB, PROAT6                |
| **行为指标**   | STUBEHA, TEACHBEHA, SCMCEG               |
| **权重体系**   | W_SCHGRNRABWT, W_FSTUWT_SCH_SUM         |


#### 二、强关联特征推测
##### 1. 学校类型与资源分配
- **SCHLTYPE (学校类型) ↔ PRIVATESCH (公私属性)**  
  数据示例：  
  - 类型`2.0`对应`private`（行0,9）  
  - 类型`3.0`对应`public`（行1,3,4,6,8,10）  
  → **假设**：类型编码直接定义公私属性

- **SCHLTYPE ↔ STRATIO (师生比)**  
  观察值对比：  
  - 私立学校（类型2.0）：`STRATIO=1.2~16.12`  
  - 公立学校（类型3.0）：`STRATIO=7.42~18.82`  
  → **假设**：公立学校师生比显著更高

##### 2. 规模连锁效应
- **SCHSIZE (学校规模) ↔ STRATIO (师生比)**  
  极端案例：  
  - 最大规模`SCHSIZE=1574`对应`STRATIO=12.69`  
  - 最小规模`SCHSIZE=75`对应`STRATIO=1.2`  
  → **假设**：学校规模与师生比呈正相关（需排除异常值）

- **SCHSIZE ↔ W_SCHGRNRABWT (学校权重)**  
  数据规律：  
  - `SCHSIZE=1572` → `W_SCHGRNRABWT=16.538`  
  - `SCHSIZE=532` → `W_SCHGRNRABWT=9.093`  
  → **假设**：大规模学校具有更高抽样权重

##### 3. 行为传导路径
- **STUBEHA (学生行为) ↔ TEACHBEHA (教师行为)**  
  数值模式：  
  - 当`STUBEHA`为负值时，`TEACHBEHA`多负（行0,2,7）  
  - 当`STUBEHA`为正值时，`TEACHBEHA`多正（行1,5）  
  → **假设**：学生行为问题会加剧教师行为压力（相关系数需计算）

- **SCMCEG (管理效率) ↔ CREACTIV (创新活动)**  
  特殊关联：  
  - `SCMCEG=0.904`时，`CREACTIV=1.0`（高频出现）  
  - `SCMCEG=-0.508`时，`CREACTIV=2.0`  
  → **假设**：管理效率与创新活动存在非线性关系

##### 4. 区域教育特征
- **Region ↔ PRIVATESCH**  
  高频区域：  
  - 区域`72413`出现2次均为私立  
  - 区域`72417`出现2次均为公立  
  → **假设**：区域代码可能反映教育政策差异

#### 三、验证优先级矩阵
| 关联对              | 验证方法                  | 预期结果方向      |
|---------------------|--------------------------|------------------|
| SCHLTYPE-PRIVATESCH | 列联表+卡方检验          | 完全对应（p≈0）  |
| SCHSIZE-STRATIO     | 散点图+Spearman相关系数  | 正相关（ρ>0.5）  |
| STUBEHA-TEACHBEHA   | 热力图+偏相关系数        | 负相关（r≈-0.6） |
| Region-PRIVATESCH   | 地区聚类分析             | 区域间显著差异   |

#### 四、深度关联假设
1. **资源分配阈值效应**  
   `RATCMP2=1.0`时，`PROAT5AB`普遍较高（行0-5,7-10均为1.0）  
   → 可能触发：资源充足时学业表现达标率跃升

2. **班级规模悖论**  
   `CLSIZE=23`的学校（高频出现）同时存在：  
   - 最高`PROAT6=0.0806`（行3）  
   - 最低`PROAT6=0.0`（行8）  
   → 班级规模对教学效果的影响存在混杂因素

3. **权重体系矛盾**  
   `W_SCHGRNRABWT`与`W_FSTUWT_SCH_SUM`增长不同步：  
   - 行2：`16.538` vs `1538.385`  
   - 行9：`9.093` vs `316.521`  
   → 学校权重与学生权重可能反映不同抽样策略

#### 最终结论
**核心关联链**：  `Region` → `PRIVATESCH` → `SCHLTYPE` → `STRATIO` → `TEACHBEHA` → `PROATCE`  

建议优先验证**区域差异对师生行为的影响路径**，使用结构方程模型(SEM)分析多级传导效应。

---

##### **Q1.d** 简述上述所得输出的特点，并比较相应推测结论与实验三的数据分析结论、T1-S4 的特征选择分析依据是否有相似之处。

$\textbf{ANS}$

1.a 的回答直接根据变量名含义推测，直接，且回答比较简洁；

1.b 的回答展示分步骤逻辑推理，保留中间假设，回答相对深入；

1.c 的回答更有深度，能够根据给出的部分数据集数据，回答更加多维全面。

相应推测结论与实验三的数据分析结论、T1-S4 的特征选择分析依据有相似之处，且逐步递进。

##### **Q2**
参考 DeepSeek 接口文档（ https://api-docs.deepseek.com/zh-cn/ ），编写代码调用 DeepSeek-V3 的 API 接口，要求大语言模型对指定问题生成 Python 代码，代码生成的提示设计举例如下：

```
[Question] 已知给定 pandas.DataFrame 实例 df，请编写一段 Python 代码输出列 A 和列 B 的平均值。
[Answer] ```print(df['A'].mean(), df['B'].mean())```
```

现需要分别求特征 `STUBEHA` 与 `TEACHBA`、`EDUSHORT` 与 `STAFFSHORT` 的相关系数。

```python
fimport requests

key = "*****************"
api_url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

question_text = """
[Question] 已知给定 pandas.DataFrame 实例 df，请分别计算以下两对列的相关系数：
1. STUBEHA 与 TEACHBEHA  
2. EDUSHORT 与 STAFFSHORT  
[Answer]
"""

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": question_text}
    ],
    # disable streaming output
    "stream": False
}

try:
    resp = requests.post(api_url, headers=headers, json=payload, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    print("API response:")
    print(reply)
except requests.exceptions.RequestException as e:
    print("Request error:", e)
```

##### **Q2.a**

请仿照上述示例设计输入提示词，调用 API 获取大语言模型的输出结果并展示。输出代码的内容和格式符合你的预期吗？代码能否正常执行（假设 df 数据集已经加载和预处理完成）？

$\textbf{ANS}$

若要在一个 pandas.DataFrame 对象 df 中计算两对字段的相关系数，可调用 corr() 方法。例如：

```python
# 计算 STUBEHA 与 TEACHBEHA 之间的皮尔逊相关系数
stu_teach_corr = df['STUBEHA'].corr(df['TEACHBEHA'])

# 计算 EDUSHORT 与 STAFFSHORT 之间的皮尔逊相关系数
edu_staff_corr = df['EDUSHORT'].corr(df['STAFFSHORT'])

print("STUBEHA vs TEACHBEHA 相关系数：", stu_teach_corr)
print("EDUSHORT vs STAFFSHORT 相关系数：", edu_staff_corr)
```
如果想使用其他相关系数（如斯皮尔曼或肯德尔），只需在 corr() 中加上 method 参数，比如 method='spearman'。

由此可见：回答比较详细，且考虑比较充分，思维较为发散。
##### **Q2.a**

尝试在上面的输入提示前补充一些样例（如上方展示的样例），重新调用 API 获取并展示大语言模型的输出。对比二者所得输出内容和格式的不同，并阐述你的发现。

$\textbf{ANS}$

```python
corr1 = df['STUBEHA'].corr(df['TEACHBEHA'])
corr2 = df['EDUSHORT'].corr(df['STAFFSHORT'])
print(corr1, corr2)
```

由此可见：当添加了一些样例后，输入结果更加接近要求，更简洁。

