# CSSIM Python 重构框架

这是连接 CSSIM UE 客户端的单环境、单事件循环 Python 算法框架，支持规则算法和自定义强化学习算法。

UE 创建算法时会把 `AlgTemplate/red` 或 `AlgTemplate/blue` 复制到 `TeamAlg/<队伍ID>`，开始仿真时生成 `TaskConfig/cssim/*.jsonc` 并把配置路径传给 `main.py`。请勿改变这三个目录的职责。

完整开发文档从 [docs/README.md](docs/README.md) 开始阅读。红蓝模板均只保留默认规则算法和精简 DQN 示例，DQN 使用固定 8 动作、固定 25 维必要态势，位于 `AlgTemplate/red/Algorithm` 与 `AlgTemplate/blue/Algorithm`。

开发验证：

```powershell
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest tests -q
```
