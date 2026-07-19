# 145 仿真工具支撑20比特及以上的量子金融算法仿真 func-145 测试结果

## 测试对象

`/home/zhenyusen/double-quant/simulator` 中软件仿真工具相关实现。

## 测试命令

```bash
.venv/bin/python tests/scripts/145-20qubit_quantum_finance_simulation.py
```

## 程序输出

```text
145 仿真工具支撑20比特及以上的量子金融算法仿真: PASS
容量检查是否通过：True
容量检查使用的后端：statevector_cpu
容量检查请求的量子比特数：20
20比特验收运行使用的后端：statevector_cpu
20比特验收运行的量子比特数：20
20比特 statevector 运行耗时（秒）：<实际耗时>
20比特 statevector 复振幅数量：1048576
20比特非零概率项数量：<实际非零概率项数量>
30比特运行使用的后端：statevector_cpu
30比特运行的量子比特数：30
30比特 statevector 运行耗时（秒）：<实际耗时>
30比特 statevector 复振幅数量：1073741824
30比特非零振幅数量：<实际非零振幅数量>
```

## 图片输出

本项为终端运行型功能测试，未生成图片输出；`images/` 目录已按验收规范保留。

## 关键结果

- 容量检查结果为 True，20 比特验收规模运行通过。
- 20 比特运行输出完整概率分布统计，statevector 长度为 $2^{20}=1,048,576$。
- 30 比特运行使用同一个 statevector CPU 后端，记录完整 statevector 长度 $2^{30}=1,073,741,824$ 和非零振幅数量。

## 输出说明

程序输出使用中文解释每个容量和 statevector 指标含义；145 同时记录 20 比特验收运行和 30 比特 statevector 运行，作为本功能验收的事实记录。
