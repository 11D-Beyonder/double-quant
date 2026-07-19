# 151 面向运维人员的编程开发接口技术报告

## 实现概述

本功能在管理员后台新增“运维接口测试”页面，用于集中测试系统运行状态、缓存链路、Python 算法环境、量子线路、量子金融算法、任务统计、管理员用户接口和证券数据查询接口。前端继续使用 React、Vite、Ant Design 和 Less，后台布局仍沿用现有 `AdminLayout`。

## 前端实现

文件：

- `frontend/src/views/Admin/OpsInterface/index.jsx`
- `frontend/src/views/Admin/OpsInterface/index.less`
- `frontend/src/api/ops.js`

路由在 `frontend/src/router/index.jsx` 中新增 `/admin/ops-interface`，菜单在 `frontend/src/layouts/AdminLayout/index.jsx` 中新增“运维接口测试”。普通主站不再暴露运维入口。

页面分为两个核心区域：

1. 接口巡检：使用表格展示接口名称、分类、方法、路径、状态、耗时和操作。每个接口都有独立测试按钮，也支持一键巡检。
2. 请求调试：通过预置接口列表填充请求方法、路径、查询参数和请求体 JSON，允许运维人员修改后发送；巡检表中的“调试”按钮会自动切换到请求调试并载入当前接口样例。

本页面不引入金融编程运行接口，也不加载 Monaco Editor、量子线路可视化或金融编程可视化工作区；金融编程能力仍由主页面已有入口承载。

## 后端接口复用

本次页面复用现有后端控制器，不新增数据库表和业务后端流程。请求调试面板统一通过现有 axios 封装调用 `/janus` 代理，保留统一 token 注入和错误处理。

## 接口清单

页面预置以下测试项：

- `GET /stock-data/status`：股票数据状态。
- `GET /stock-data/securities`：股票池分页样例。
- `GET /stock-data/jobs`：数据导入记录。
- `GET /redis/check`：Redis 兼容性检查。
- `GET /quantum/finance/test-python`：Python 算法环境。
- `POST /quantum/validate-qasm`：QASM 语法校验。
- `POST /quantum/run-qasm`：QASM 运行冒烟。
- `GET /quantum/finance/circuit-config`：量子电路配置。
- `POST /quantum/finance/run`：期权定价冒烟。
- `GET /record/count`：任务统计。
- `GET /record/query/page`：任务分页查询。
- `GET /admin/users`：管理员用户列表。
- `GET /security/search`：证券搜索样例。

## 安全设计

- 页面入口位于 AdminLayout，前端会根据 token 和用户角色拦截非管理员访问。
- 后端对 `/admin/**` 使用 `hasRole("ADMIN")`。
- 请求调试面板只通过现有 axios 封装调用 `/janus` 代理，保留统一 token 注入和错误处理。
