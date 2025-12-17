# 软件可靠性分析平台 · 需求分析文档


## 1. 背景与目标
- 为测试、运维团队提供统一的可靠性数据入口、模型分析与可视化决策面板。
- 支持手动、CSV/Excel、MySQL 等多源数据导入，结合 DeepSeek 智能列映射降低数据准备成本。
- 集成经典模型（Goel-Okumoto、JM、Musa-Okumoto、Crow-AMSAA、Duane）、人工智能模型（BPN、RBF、SVM、GEP）、时间序列、组合加权、SDA 场景模型与模型对比。
- 输出趋势、预测与对比结果，可一键导出 HTML 报告，便于汇报与留存。

## 2. 范围
1) 数据导入与预处理：文件上传、MySQL 拉取；弹窗中样本预览、本地映射或 DeepSeek 映射确认后写入。  
2) 模型分析：各板块需点击“开始分析”才计算，避免导入后自动耗时。  
3) 可视化与公式：Chart.js 图表、MathJax 公式面板。  
4) 导出：当前模型参数、最近记录、图表与公式一键导出 HTML。  
5) 系统管理：用户新增/列表（Cloudflare Worker KV），状态标记。  

## 3. 术语
- MTBF：Mean Time Between Failure，平均失效间隔。  
- PKR：Probability of Keeping Reliability，保持目标可靠度的概率。  
- SDA：Scenario-Driven Assurance，场景驱动的可靠性保障。  

## 4. 角色与权限
| 角色 | 目标 | 权限 |
| --- | --- | --- |
| 可靠性研究员 | 维护模型与参数、验证曲线 | 导入数据、运行分析、导出报告 |
| 数据分析员 | 监控与汇报 | 查看全部可视化、导出报告 |
| 运维/测试经理 | 对比决策 | 运行各模型、查看 PKR/性能对比 |
| 访客/审计 | 只读 | 查看仪表盘与需求文档 |

## 5. 用例概述

![](image.png)

## 6. 业务流程（文件导入示例）

![](image-1.png)

## 7. 功能性需求（细化）
1) 数据导入  
   - 支持手动录入、CSV/Excel 上传、MySQL 连接（DSN 来自 `.env`）。  
   - 获取样本后弹出映射弹窗：表格预览、本地映射或 DeepSeek 推荐映射，用户确认后写入。  
   - 导入成功后停留在导入页，不自动分析。  
2) 模型分析（按板块触发）  
   - 经典：Goel-Okumoto、JM、Musa-Okumoto、Crow-AMSAA、Duane。  
   - 人工智能：BPN、RBF、SVM、GEP。  
   - 时间序列：ARIMA、Holt-Winters。  
   - 组合加权：静态/动态权重展示。  
   - SDA：场景路径可靠度。  
   - 模型对比：PKR 雷达、性能/成本柱状。  
   - 点击“开始分析”后调用 `/api/analyze/<section>` 返回图表与公式（MathJax）。  
3) 可视化与公式  
   - Chart.js 支持折线、柱状、雷达等；公式面板与模型说明并列展示。  
   - 防止 NaN/Inf：后端清洗预测数据，保证 JSON 可序列化。  
4) 导出  
   - `GET /api/export/html`：导出当前仪表盘（参数、记录、图表、公式）为独立 HTML；前端“导出分析 HTML”按钮下载。  
5) 系统管理  
   - 用户创建/列表，数据存放 Cloudflare Worker KV（`WORKER_ENDPOINT`）；状态显示启用/禁用。  

## 8. 数据与接口
- 核心字段：`module`、`failures`、`mtbf`、`runtime`、`timestamp`。  
- 接口概要：  
  - `POST /api/import/file/sample`：获取文件样本  
  - `POST /api/import/file`：确认写入  
  - `POST /api/import/mysql/sample`：获取 MySQL 样本  
  - `POST /api/import/mysql`：确认写入  
  - `POST /api/deepseek/mapping`：AI 列映射  
  - `GET /api/analyze/<section>`：按板块返回图表与公式  
  - `GET /api/export/html`：导出报告  
  - `GET/POST /api/users`：用户管理  

## 9. UML 视图
### 9.1 组件/层次关系（简化）

![](image-2.png)

### 9.2 活动图（“开始分析”）

![](image-3.png)

## 10. 界面与交互要点
- 左侧导航切换板块，默认展开“数据导入”；右侧内容区按板块折叠展示。  
- 导入弹窗：表格预览、映射选择、DeepSeek 推荐回显，用户确认后写入。  
- 每个模型板块按钮“开始分析”才触发计算；下方公式面板与图表对应。  
- “导出分析 HTML”按钮生成当前全局数据的离线报告。  

## 11. 非功能性要求
- 易用性：弹窗式映射、表格预览，操作提示清晰；支持中文界面。  
- 性能：分析按需触发；Chart.js 前端渲染，避免后端阻塞。  
- 可靠性：导入/映射失败有明确提示；预测数据清洗防止 NaN/Inf。  
- 安全：API Key、DSN 存放 `.env`，不写死代码；仅本地存储演示，无强制鉴权。  
- 可维护性：模块化（数据存储、可靠性服务、DeepSeek 服务、前端）；可扩展新模型或新导入源。  
- 可观测性：导入日志、状态提示；控制台可查看关键请求与响应。  

## 12. 验收标准
- 导入链路可用：文件/数据库能取样本，映射确认后写入；导入日志实时更新。  
- 各板块点击“开始分析”后返回图表与公式；MathJax 正常渲染。  
- 导出 HTML 可离线打开，包含参数、记录、图表与公式。  
- 用户管理可新增并展示状态（Worker KV）。  
