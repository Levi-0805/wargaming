# TeamAlg 队伍目录说明

`TeamAlg` 保存 UE 本轮实际加载的队伍算法。`RedModel` 和 `BlueModel` 是 UE 未选择算法时使用的两个默认占位队伍；UE 创建的实际选手队伍也会出现在本目录中：

| 目录 | 阵营 | 用途 |
|---|---|---|
| `RedModel` | 红方 | 为本方实体返回`Idle`，作为未选择红方算法时的默认入口。 |
| `BlueModel` | 蓝方 | 为本方实体返回`Idle`，作为未选择蓝方算法时的默认入口。 |

UE创建算法时，会把`AlgTemplate/red`或`AlgTemplate/blue`复制到
`TeamAlg/<teamID>`，运行时通过以下路径加载：

```text
TeamAlg.<teamID>.Algorithm.RLAgentAlgorithm->ReinforceAgentAlgorithm
```

`RedModel` 和 `BlueModel` 是 UE 配置所需的兜底入口，必须保留。其他随机 ID 目录由 UE 创建并属于对应选手。

## 用户算法配置

UE 生成的 TaskConfig 只指向各队伍的算法入口和 `AlgorithmConfig`，用户不应修改或依赖 TaskConfig 保存算法参数。规则/DQN 模式、网络参数和是否加载模型应配置在自己的文件中：

```text
TeamAlg/<teamID>/Algorithm/RLAgentAlgorithm.py
```

模板 DQN 从头训练时设置 `mode="reinforcement_learning"`、`load_model=False`；继续训练或比赛加载已有模型时设置 `load_model=True`。检查点固定保存到并加载自：

```text
TeamAlg/<teamID>/TrainLogs/dqn_model.pt
```

比赛时 UE 还必须下发 `training=0`，用于禁止探索、经验写入、网络更新和模型覆盖。`load_model=True` 但检查点不存在时，Python 会在启动阶段明确报错。
