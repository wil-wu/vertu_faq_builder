# Vertu Algorithm Service

基于 FastAPI 的现代化算法服务模板,支持多服务架构、API 版本化、Docker 部署和可观测性。

## 🚀 特性

- ✅ **UV 包管理**: 使用 uv 进行快速依赖管理
- ✅ **FastAPI 框架**: 高性能异步 Web 框架
- ✅ **模块化架构**: APIRouter 组织的子服务结构
- ✅ **API 版本化**: 支持多版本 API 并存
- ✅ **面向对象设计**: 核心模块采用 OOP 范式
- ✅ **配置管理**: Pydantic Settings 管理全局和服务配置
- ✅ **共享对象**: 应用级和服务级共享对象管理
- ✅ **线程安全**: 单例模式确保多线程安全
- ✅ **可观测性**: Prometheus 指标监控
- ✅ **Docker 支持**: 完整的容器化方案
- ✅ **测试框架**: Pytest 异步测试支持

## 📁 项目结构

```
algorithm-service/
├── app/                          # 主应用
│   ├── app.py                    # 应用入口
│   ├── config.py                 # 全局配置
│   ├── scanner.py                # 路由自动扫描
│   ├── core/                     # 核心模块
│   │   ├── exceptions.py         # 自定义异常
│   │   └── middlewares.py        # 中间件
│   └── services/                 # 子服务
│       └── answer_enhancement    # 答案增强服务
│           ├── checkers.py       # 策略检查器
│           ├── config.py         # 服务配置
│           ├── deps.py           # 依赖注入
│           ├── enhancers.py      # 定向策略增强器
│           ├── enum.py           # 策略枚举
│           ├── models.py         # Pydantic 模型
│           ├── router.py         # API 路由
│           └── service.py        # 业务逻辑
└── tests/                        # 测试
    ├── conftest.py               # 测试配置
    └── services/                 # 服务测试
```

## 🛠️ 快速开始

### 1. 安装依赖

```bash
# 使用 uv 初始化项目
uv init

# 安装依赖
uv sync

# 或者安装开发依赖
uv sync --dev
```

### 2. 配置环境变量

```bash
# 编辑 .env 文件，按需配置
vim .env
```

### 3. 运行服务

#### 本地开发

```bash
# 运行 API 服务
uv run main.py

# 或
uvicorn main:app --reload
```

### 4. 访问服务

- API 文档: http://localhost:8000/docs
- Prometheus 指标: http://localhost:8000/metrics
- Prometheus UI（后续部署）: http://localhost:9090

## 📝 创建新服务

### 1. 创建服务目录

```bash
mkdir -p app/services/your_service
cd app/services/your_service
```

### 2. 创建必要文件

```bash
config.py
deps.py
models.py
router.py
service.py
```
### 3. 自动注册

路由会被 `Scanner` 自动发现并注册,无需手动配置。

## 🧪 测试
补充

## 📊 监控

### Prometheus 指标

服务自动暴露以下指标:

- `http_requests_total`: 请求总数
- `http_request_duration_seconds`: 请求耗时
- `http_requests_inprogress`: 进行中的请求数

### 健康检查

```bash
# 主服务健康检查
curl http://localhost:8000/health
```

## 🔧 配置说明

### 全局配置 (app/config.py)

存放一些公共配置

### 服务配置

每个服务都有独立的配置文件,使用环境变量前缀隔离

## 🔒 线程安全

所有共享对象都使用线程安全的单例模式

## 📦 依赖管理

```bash
# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name

# 更新依赖
uv sync --upgrade

# 查看依赖树
uv tree
```

## 🎯 最佳实践

1. **配置管理**: 使用 Pydantic Settings,环境变量优先
2. **日志记录**: 使用结构化日志,包含上下文信息
3. **错误处理**: 在服务层捕获异常,返回标准响应
4. **依赖注入**: 使用 FastAPI 依赖注入系统
5. **异步优先**: 所有 I/O 操作使用异步
6. **线程安全**: 共享对象使用单例模式 + 锁
7. **测试覆盖**: 保持 80%+ 测试覆盖率

## 🐛 故障排查
补充