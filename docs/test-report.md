# 软件可靠性分析平台测试报告（简要）

## 1. 测试环境
- 运行方式：`python app.py`（Flask 开发模式，127.0.0.1:5000）
- 依赖：Anaconda Python，前端使用内置资源（assets/js、css），DeepSeek 映射可选。
- 数据：本地手工/文件导入，MySQL 可选；Workers KV 用于用户管理。

## 2. 覆盖范围与用例
### 2.1 数据导入
- 手动导入：填写模块/失效/MTBF/时长，提交后列表与图表刷新（HTTP 200）。
- 文件导入（CSV）：上传示例文件，预览样本→DeepSeek 映射→应用映射→成功写入（HTTP 200）。  
  - 失败用例：不含数据行，返回 400 “文件内容为空或无法解析”。
- MySQL 导入：校验表名/增量字段合法性；连接缺失字段返回 400，合法连接未实测（需真实 DB）。

### 2.1.1 控制台日志（文件导入示例）
- “POST /api/import/file/sample” → 200，DeepSeek 映射返回 `{'module': 'module', 'failures': 'failures', 'mtbf': 'mtbf', 'runtime': 'runtime'}`  
- “POST /api/import/file” → 若数据异常曾触发 `_rmse` 溢出，现已加入裁剪防护；正常情况下应 200 并刷新仪表盘。

### 2.2 模型加载
- 导入后默认加载仪表盘：经典模型 5 图、AI 4 图、时间序列 2 图、组合 2 图、SDA/对比 2 图（共 16）。
- 按需分析：`/api/analyze/<section>` 返回对应板块 JSON（classic/ai-models/time-series…），状态 200。

### 2.3 DeepSeek 映射
- 调用 `/api/deepseek/mapping` 传入表头+样本行，返回字段映射（HTTP 200）。  
- 失败用例：缺少 headers/rows 返回 400。

### 2.4 用户管理（Cloudflare Worker）
- 列表：`GET /api/users` 成功返回 KV 中成员。
- 新增：前端表单提交 `/api/users`，云端返回 success 后表格追加。
- 删除：可通过 Worker `DELETE /users/<id>`；UI 层暂未添加按钮（可在后续完善）。

### 2.5 本地模型生成与摘要（脚本执行）
- 对 `experiments/reliability_eval.csv` 调用 `build_chart_payload`：`charts_count=16`，首个图表 `chart-goel`，AI 图表 ID 为 `chart-bpn/rbf/svm/gep`。  
- 归一化样本数：60；样本示例：`{'module': 'Gamma', 'failures': 3, 'mtbf': 262.0, 'runtime': 180.0, 'timestamp': '2025-01-01T00:00:00', 'source': 'unknown'}`。  
- 导入统计：`{'daily': 0, 'latest_source': 'unknown'}`；模型参数示例：Goel-Okumoto α≈0.242, β=0.02, JM α≈0.4, β≈0.151，时间戳为实时生成。

## 3. 结果与问题
- 结果：手动/文件导入、仪表盘渲染、DeepSeek 映射、用户云端同步均通过基本验证；按需分析接口返回正常。
- 问题：  
  1) RMSE 计算曾因极端值溢出导致 500，已在 `_rmse` 增加裁剪保护；若数据异常仍可能需要更严格的清洗。  
  2) MySQL 导入仅做连通性/参数校验演练，未覆盖真实业务库的字段/权限/大批量性能。  
  3) 用户删除通路可用（Worker DELETE），但前端未提供入口，需补充 UI 与二次确认。

## 4. 改进建议
- 增加自动化测试（pytest/flask testclient）覆盖导入、映射、`/api/analyze`、RMSE 裁剪分支。
- MySQL 集成测试：准备小型测试库，验证增量字段、大批量导入及权限控制。
- 前端：用户管理增加删除按钮与二次确认；模型加载增加超时/错误提示；文件导入增加异常值提示。
- 模型：对高维/异常数据引入标准化与异常检测，避免溢出；为 AI 模型加早停与参数网格搜索（可选）。
