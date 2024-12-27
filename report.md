### Operator Time Consuming Ranking Table

| OpType             | CallNumber | CPUTime(us) | GPUTime(us) | NPUTime(us) | TotalTime(us) | TimeRatio(%) |
|--------------------|------------|-------------|-------------|-------------|---------------|--------------|
| ConvExSwish         | 138        | 0           | 0           | 80201       | 80201         | 76.94%       |
| Concat              | 38         | 0           | 0           | 5868        | 5868          | 5.63%        |
| Split               | 15         | 0           | 0           | 5270        | 5270          | 5.06%        |
| AveragePool         | 5          | 0           | 0           | 4203        | 4203          | 4.03%        |
| exSoftmax13         | 1          | 0           | 0           | 2169        | 2169          | 2.08%        |
| MaxPool             | 8          | 0           | 0           | 2159        | 2159          | 2.07%        |
| Add                 | 18         | 0           | 0           | 1237        | 1237          | 1.19%        |
| Conv                | 7          | 0           | 0           | 771         | 771           | 0.74%        |
| Resize              | 2          | 0           | 0           | 663         | 663           | 0.64%        |
| Reshape             | 5          | 568         | 0           | 6           | 574           | 0.55%        |
| Transpose           | 2          | 0           | 0           | 486         | 486           | 0.47%        |
| Sigmoid             | 1          | 0           | 0           | 354         | 354           | 0.34%        |
| Mul                 | 2          | 0           | 0           | 148         | 148           | 0.14%        |
| Sub                 | 2          | 0           | 0           | 99          | 99            | 0.09%        |
| OutputOperator      | 1          | 30          | 0           | 0           | 30            | 0.03%        |
| InputOperator       | 1          | 4           | 0           | 0           | 4             | 0.00%        |

### Total Inference Time
- **Total Inference Time (in milliseconds, 4 decimal places):** 113.4130 ms

### Summary and Analysis

1. **Dominant Operators:**
   - The operator `ConvExSwish` is the most time-consuming, accounting for **76.94%** of the total inference time. This indicates that the model spends the majority of its time in convolutional operations followed by a swish activation function.
   - The `Concat` and `Split` operators also consume a significant portion of the inference time, with **5.63%** and **5.06%** respectively.

2. **Less Time-Consuming Operators:**
   - Operators like `Reshape`, `Transpose`, and `Sigmoid` consume relatively less time, each contributing less than **1%** of the total inference time.
   - The `OutputOperator` and `InputOperator` are the least time-consuming, with **0.03%** and **0.00%** respectively.

3. **CPU vs. NPU Utilization:**
   - The majority of the inference time is spent on the NPU, with **103,634 microseconds** (**103.634 milliseconds**) spent on NPU operations.
   - The CPU time is minimal, with only **602 microseconds** (**0.602 milliseconds**) spent on CPU operations.

4. **Potential Optimization Areas:**
   - Given that `ConvExSwish` is the most time-consuming operator, optimizing the convolutional layers or exploring alternative activation functions could potentially reduce the inference time.
   - The `Concat` and `Split` operators, while not as dominant as `ConvExSwish`, still contribute a significant portion of the inference time. Optimizing these operations could also lead to performance improvements.

5. **Overall Performance:**
   - The total inference time of **113.4130 milliseconds** indicates that the model is performing inference within a reasonable time frame for many real-time applications. However, further optimizations could reduce this time, making the model even more suitable for latency-sensitive applications.

### Conclusion
The inference time is primarily dominated by convolutional operations (`ConvExSwish`), followed by concatenation (`Concat`) and splitting (`Split`) operations. The model efficiently utilizes the NPU, with minimal CPU involvement. To further optimize performance, focusing on reducing the time spent in convolutional layers and concatenation/splitting operations could yield significant improvements.


### 操作耗时排名表

| 操作类型             | 调用次数 | CPU时间(us) | GPU时间(us) | NPU时间(us) | 总时间(us) | 时间比例(%) |
|--------------------|------------|-------------|-------------|-------------|---------------|--------------|
| ConvExSwish         | 138        | 0           | 0           | 80201       | 80201         | 76.94%      |
| Concat              | 38         | 0           | 0           | 5868        | 5868          | 5.63%       |
| Split               | 15         | 0           | 0           | 5270        | 5270          | 5.06%       |
| AveragePool         | 5          | 0           | 0           | 4203        | 4203          | 4.03%       |
| exSoftmax13         | 1          | 0           | 0           | 2169        | 2169          | 2.08%       |
| MaxPool             | 8          | 0           | 0           | 2159        | 2159          | 2.07%       |
| Add                 | 18         | 0           | 0           | 1237        | 1237          | 1.19%       |
| Conv                | 7          | 0           | 0           | 771         | 771           | 0.74%       |
| Resize              | 2          | 0           | 0           | 663         | 663           | 0.64%       |
| Reshape             | 5          | 568         | 0           | 6           | 574           | 0.55%       |
| Transpose           | 2          | 0           | 0           | 486         | 486           | 0.47%       |
| Sigmoid             | 1          | 0           | 0           | 354         | 354           | 0.34%       |
| Mul                 | 2          | 0           | 0           | 148         | 148           | 0.14%       |
| Sub                 | 2          | 0           | 0           | 99          | 99            | 0.09%       |
| OutputOperator      | 1          | 30          | 0           | 0           | 30            | 0.03%       |
| InputOperator       | 1          | 4           | 0           | 0           | 4             | 0.00%        |

### 总推理时间
- **总推理时间（以毫秒为单位，保留四位小数）：** 113.4130 ms

### 总结与分析

1. **主要操作：**
   - 操作 `ConvExSwish` 是最耗时的，占总推理时间的 **76.94%**。这表明模型大部分时间都花在卷积操作后跟一个 swish 激活函数上。
   - `Concat` 和 `Split` 操作也消耗了相当一部分推理时间，分别为 **5.63%** 和 **5.06%**。

2. **耗时较少的操作：**
   - 像 `Reshape`、`Transpose` 和 `Sigmoid` 这样的操作消耗的时间相对较少，每个对总推理时间的贡献都不到 **1%**。
   - `OutputOperator` 和 `InputOperator` 是最不耗时的，分别为 **0.03%** 和 **0.00%**。

3. **CPU与NPU利用率：**
   - 大部分推理时间都花在了NPU上，**103,634 微秒**（**103.634 毫秒**）用于NPU操作。
   - CPU时间非常少，只有 **602 微秒**（**0.602 毫秒**）用于CPU操作。

4. **潜在优化领域：**
   - 鉴于 `ConvExSwish` 是最耗时的操作，优化卷积层或探索替代激活函数可能会减少推理时间。
   - `Concat` 和 `Split` 操作虽然不如 `ConvExSwish` 占主导地位，但仍占相当一部分推理时间。优化这些操作也可能导致性能提升。

5. **整体性能：**
   - 总推理时间为 **113.4130 毫秒**，表明模型在许多实时应用中进行推理的时间框架是合理的。然而，进一步的优化可以减少这个时间，使模型更适合对延迟敏感的应用。

### 结论
推理时间主要由卷积操作（`ConvExSwish`）主导，其次是连接（`Concat`）和分割（`Split`）操作。模型有效地利用了NPU，CPU参与度最小。为了进一步优化性能，专注于减少卷积层和连接/分割操作所花费的时间可能会带来显著的改进。

### Model size: 27.7908 MB
### SDK API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
### Driver Version: 0.9.8

Total Operator Elapsed Per Frame Time(us): 131201
Total Memory Read/Write Per Frame Size(KB): 235155.92
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
                                 Operator Time Consuming Ranking Table
---------------------------------------------------------------------------------------------------
OpType             CallNumber   CPUTime(us)  GPUTime(us)  NPUTime(us)  TotalTime(us)  TimeRatio(%)
---------------------------------------------------------------------------------------------------
ConvExSwish        138          0            0            95657        95657          72.91%
Concat             38           0            0            10339        10339          7.88%
Split              15           0            0            6891         6891           5.25%
AveragePool        5            0            0            4817         4817           3.67%
Add                18           0            0            3539         3539           2.70%
MaxPool            8            0            0            3101         3101           2.36%
exSoftmax13        1            0            0            2344         2344           1.79%
Conv               7            0            0            1168         1168           0.89%
Reshape            5            1065         0            7            1072           0.82%
Resize             2            0            0            845          845            0.64%
Transpose          2            0            0            626          626            0.48%
Sigmoid            1            0            0            398          398            0.30%
Sub                2            0            0            202          202            0.15%
Mul                2            0            0            150          150            0.11%
OutputOperator     1            38           0            0            38             0.03%
InputOperator      1            14           0            0            14             0.01%
---------------------------------------------------------------------------------------------------
Total                           1117         0            130084       131201
---------------------------------------------------------------------------------------------------

Total inference time (in milliseconds, 4 decimal places): 151.6830 ms