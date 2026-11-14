# CSSIM Python 重构框架

这是连接 CSSIM UE 客户端的单环境、单事件循环 Python 算法框架，支持规则算法和自定义强化学习算法。

UE 创建算法时会把 `AlgTemplate/red` 或 `AlgTemplate/blue` 复制到 `TeamAlg/<队伍ID>`，开始仿真时生成 `TaskConfig/cssim/*.jsonc` 并把配置路径传给 `main.py`。请勿改变这三个目录的职责。

完整开发文档从 [docs/README.md](docs/README.md) 开始阅读。红蓝模板均只保留默认规则算法和精简 DQN 示例，DQN 使用固定 8 动作、固定 25 维必要态势，位于 `AlgTemplate/red/Algorithm` 与 `AlgTemplate/blue/Algorithm`。

手动运行：先在CSSIM选择场景、红蓝算法和局数，进入等待Python连接的状态；然后直接运行 `main.py`，在控制台输入：

- `1`：历史高分基线＋已证实的HQ留守/回点修复（默认；分数收益仍需实战验证）。
- `2`：未加守备覆盖层的原始高分基线，用于对照。
- `3`：平台原配置。

随后选择本次局数，不得超过平台配置。现有PyCharm `main` 中指向 `iql_eval_red.jsonc` 的旧参数也会进入该选择界面；当前平台生成的连接端口和局数优先，静态三局模板不会限制为三局。高分模式逐文件校验历史快照和原模型，并写独立有效配置，不修改平台输入。已有main运行时，第二个main会明确退出。

命令行可用 `--strategy hq-guard` 或 `--strategy historical` 明确选择；`-c ... --non-interactive` 保留原配置。带批次/源码标识或平台 `-s` 的托管启动不显示菜单，也不允许临时改变已绑定策略。手动运行仍如实记录未由启动器绑定的整套源码来源，不补造Git提交。

开发验证：

```powershell
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest tests -q
```
