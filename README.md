# 软件可靠性分析平台（Software Reliability Analysis Platform）

一个面向软件可靠性数据分析与建模的可视化平台，支持上传/管理可靠性数据、进行可靠性模型拟合与对比分析，并生成相关报告与图表，帮助用户评估软件失效率、可靠度趋势与预测结果。

## ✨ 功能特性


- 📊 **可靠性数据导入与预处理**
  - 支持 CSV 数据导入（示例：`reliability_sample_data.csv`）
  - 自动识别关键字段、时间/失效次数数据格式化等处理
- 📈 **软件可靠性建模与拟合**
  - 支持常见软件可靠性增长模型（如 GO、JM 等，可扩展）
  - 参数估计、拟合优度评估、对比分析
- 🧠 **实验与算法扩展**
  - `experiments/` 目录用于训练/实验脚本管理
  - 可支持不同算法或模型的扩展试验
- 🖥️ **Web 平台可视化**
  - 通过 `templates/` + Web 服务提供交互式界面
  - 自动生成图表与报告模板
- 📑 **报告与模板支持**
  - `软件配套模板/` 中提供报告/文档模板
  - 可快速生成实验报告或模型分析总结

---

## 📂 项目结构

```bash
Software-Reliability-Analysis-Platform/
├── app.py                     # 主应用入口（Web 服务/路由）
├── serve.py                   # 启动脚本/服务封装
├── requirements.txt           # Python 依赖
├── reliability_sample_data.csv# 示例数据
├── services/                  # 后端服务模块（数据处理/模型计算）
├── templates/                 # Web 前端模板（HTML）
├── assets/                    # 静态资源（CSS/JS/图片）
├── docs/                      # 项目说明与文档
├── experiments/               # 实验脚本与算法验证
├── png/                       # 可视化结果图片
└── 软件配套模板/              # 项目配套报告模板等
````

---

## 🛠️ 环境要求

* Python 3.8+
* 推荐使用虚拟环境（venv / conda）

---

## 🚀 快速开始

### 1）克隆项目

```bash
git clone https://github.com/jiwei741/Software-Reliability-Analysis-Platform.git
cd Software-Reliability-Analysis-Platform
```

### 2）创建虚拟环境并安装依赖

#### 使用 venv

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 使用 conda

```bash
conda create -n reliability python=3.9 -y
conda activate reliability
pip install -r requirements.txt
```

---

### 3）运行项目

项目可通过 `app.py` 启动：

```bash
python app.py
```

启动成功后，浏览器访问：

```
http://127.0.0.1:5000
```

> 如果端口不同，请根据控制台输出调整。

---

## 📊 示例数据格式

项目自带示例数据：

* `reliability_sample_data.csv`
---

## 🔬 模型说明

平台可用于实现/扩展的软件可靠性增长模型包括但不限于：

* GO 模型（Goel-Okumoto NHPP）
* JM 模型（Jelinski-Moranda）
* Musa 模型
* S 型模型（Yamada S-shaped）

你可以在 `services/` 中扩展模型实现，在 `experiments/` 中进行对比试验。

---

## 📑 结果输出

平台可输出：

* 拟合曲线（失效累计/强度/可靠度）
* 参数估计结果（α、β 等）
* 拟合优度指标（MSE、R²、AIC/BIC 等）
* 自动生成报告（配套模板在 `软件配套模板/`）

---

## 🤝 贡献方式

欢迎提交 Issue / PR：

1. Fork 本仓库
2. 创建分支：`git checkout -b feature/xxx`
3. 提交代码：`git commit -m "add xxx"`
4. 发起 Pull Request

---

## 📜 License

本项目仅用于学习与科研用途。

---

## 👨‍💻 作者

* **jiwei741**、**Valerius-Astoria**
* 若你觉得项目对你有帮助，欢迎 Star ⭐

