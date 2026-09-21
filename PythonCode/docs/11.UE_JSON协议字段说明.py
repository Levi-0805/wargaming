# -*- coding: utf-8 -*-
"""CSSIM UE ↔ Python 底层交互 JSON 说明（2026-09-07 实测版）
================================================================

本轮事实：红蓝双方均为规则模式，1 回合、200 次 step、70 个战斗实体、2 个
指挥员、1 个关键目标；最后 ``timeCnt=199``，蓝方获胜，结束原因
``RoundMaxCntReached``。Python 共发送 14,000 条动作字符串，均满足协议基本格式。

底层不是“直接发送 JSON 的 TCP”：外层是 gRPC 双向流 StreamMessage，JSON 文本
以 UTF-8 bytes 放在 prepare/frame/custom 的 data 字段中。当前调用完全串行。

交互顺序：

    建立 gRPC DataChannel
      -> handshake(protocol_version=0)
      -> 循环 prepare(reset_property)，直到 UE 点击开始并返回 propertyArr
      -> prepare(reset_obs)，取得初始 dataArr/dataGlobal（本轮 timeCnt=-1）
      -> Python 决策并 frame(step)，取得下一帧（-1 -> 0 -> ... -> 199）
      -> 最后一帧 episodeDone=true，并带 EndEpisode 事件
      -> UE 另发 finish 消息（计分展示数据）
      -> Python custom(end_unreal_engine)，关闭 gRPC 流

"""


# ============================================================================
# 一、gRPC 外层封装（JSON 位于 data 字节字段内）
# ============================================================================

GRPC_ENVELOPES = {
    "handshake": {
        "info": {"id": "py-<随机12位>", "name": "cssim-<随机6位>"},
        "handshake": {"protocol_version": 0},
    },
    "prepare_request": {
        "info": {"id": "py-<本连接ID>", "name": "cssim-<本连接名称>"},
        "prepare": {"type": "reset_property 或 reset_obs", "data": b"<UTF-8 JSON>"},
    },
    "prepare_response": {
        "prepare": {
            "type": "可能为空；不能作为有效回包的强制匹配条件",
            "data": b"<UTF-8 JSON: propertyArr or dataArr/dataGlobal>",
        },
    },
    "frame": {
        "info": {"id": "py-<本连接ID>", "name": "cssim-<本连接名称>"},
        "frame": {"data": b"<UTF-8 JSON>", "sleepTime": 2.0},
    },
    "custom": {
        "info": {"id": "py-<本连接ID>", "name": "cssim-<本连接名称>"},
        "custom": {"type": "end_unreal_engine", "data": b"<UTF-8 JSON>"},
    },
    "finish": {
        "finish": {"type": "<UE提供>", "data": "<UTF-8 bytes：计分展示 JSON>"},
    },
}


# ============================================================================
# 二、Python -> UE：JSON 请求
# ============================================================================

RESET_PROPERTY_REQUEST = {
    "valid": True,
    "DataCmd": "reset_property",
    "NumAgents": 0,          # 0：实体数量由 UE 场景决定
    "TimeStepMax": 200,      # 配置 MaxStepsPerEpisode
    "TimeDilation": 1.0,     # UE 仿真时间倍率
    "FrameRate": 60.0,       # UE 配置帧率
    "TimeStep": 0,
    "Actions": None,         # 旧接口占位；当前动作使用 StringActions
}

# 准备阶段采用超时重试，持续发送 reset_property，直到 UE 点击开始并返回有效
# propertyArr；这是正常等待机制，不是重复创建回合。
# 2026-09-14 实跑确认：UE 可在 prepare.type 为空时返回有效 propertyArr。客户端应保持
# prepare 请求串行并校验 data 中的 JSON 必需字段，不能要求响应回显请求 type。

RESET_OBS_REQUEST = {
    **RESET_PROPERTY_REQUEST,
    "DataCmd": "reset_obs",
}

# 初始回包 timeCnt=-1，所以第一条 step 请求的 TimeStep 也是 -1；UE 回包 timeCnt=0。
# 最后一条请求 TimeStep=198，UE 回包 timeCnt=199。
STEP_REQUEST = {
    "valid": True,
    "DataCmd": "step",
    "TimeStep": 10,          # 始终使用产生本次决策的上一回包 timeCnt
    "Actions": None,
    "StringActions": [       # 每个战斗实体一条；UE按字符串末尾执行者UID定位，不按数组下标定位
        "ActionSet2::Moving;;Points=[19585.14,33011.71,1152.0];344",
        "ActionSet2::NormalAttacking;274;56",
        "ActionSet2::Guard;;Points=[20585.14,33011.71,1152.0];354",
        "ActionSet2::Idle;N/A;51",
    ],
    "RSVD1": "None",        # 当前固定值；与回包 rSVD1 不是同一用途
}

STEP_REQUEST_FIELDS = {
    "valid": "请求有效标志；当前固定 true。",
    "DataCmd": "业务指令：step。",
    "TimeStep": "本次动作对应的上一帧计数。",
    "Actions": "旧数值动作接口占位，当前为 null。",
    "StringActions": (
        "UE 标准动作字符串数组；框架统一补执行者动态 UID。UE 按每条字符串中的执行者 UID "
        "定位单位，数组顺序无需与 propertyArr 原始顺序一致。"
    ),
    "RSVD1": "step 请求预留字符串，当前固定为 'None'。",
}

END_REQUEST = {"valid": True, "DataCmd": "end_unreal_engine"}


# ============================================================================
# 三、UE -> Python：reset_property 响应
# ============================================================================

# 本轮 propertyArr 共 72 条：70 个战斗实体 + 红蓝各 1 个指挥员。
# UID 是 UE 本局动态标识，不得写死。下面保留本轮真实字段并用一名士兵作样例。
RESET_PROPERTY_RESPONSE = {
    "valid": True,
    "propertyArr": [{
        "className": "士兵_1",                # 场景显示名称
        "agentTeam": 0,                       # 0=红方，1=蓝方
        "indexInTeam": 201,                   # 当前指挥员 UID，不是队内数组下标
        "uId": 56,                            # 本局全局唯一 UID
        "debugAgent": False,
        "maxMoveSpeed": 0,                    # 本轮全为 0；实体仍移动，当前不可依赖
        "initLocation": {"x": 38164.81, "y": 22307.22, "z": 641.9884},
        "initRotation": {"x": -0.005493164, "y": 0.005493164, "z": 179.9945},
        "initRotator": {"pitch": 0.005493164, "yaw": 179.9945, "roll": -0.005493164},
        "agentScale": {"x": 0, "y": 0, "z": 0},  # 动态回包通常为 1
        "initVelocity": {"x": 0, "y": 0, "z": 0},
        "agentHp": 100,
        "weaponCD": 0.2,
        "bulletCnt": 250,
        "isTeamReward": False,
        "type": "BP_BaseSoldier_C",          # 稳定技术类型
        "weaponType": "",
        "color": "",
        "dodgeProb": 0,
        "explodeDmg": 20,
        "fireRange": 1000,                    # 攻击射程，位置同单位（通常为 cm）
        "guardRange": 1400,
        "perceptionRange": 10000,             # 感知范围，不是 fireRange
        "rSVD1": "",
        "rSVD2": "",
    }],
}

PROPERTY_FIELDS = {
    "className": "中文场景名称，如士兵_1、无人机_0、指挥员_0。",
    "agentTeam": "阵营编号：0 红、1 蓝；其它阵营会被当前框架过滤。",
    "indexInTeam": (
        "当前单位所连接的指挥员 UID。本轮红方为 201、蓝方为 205；指挥员可在运行中切换，"
        "也可以由士兵担任，因此不能把它当作实体队内序号。"
    ),
    "uId": "UE 每局动态全局 UID；动作目标和执行者都使用它。",
    "type": "UE 蓝图/逻辑类型，是算法判断兵种的主要字段。",
    "initLocation/initVelocity": "开局位置与速度三维向量。",
    "initRotation": "x/y/z 表示法的初始旋转；具体轴语义依 UE。",
    "initRotator": "pitch/yaw/roll 初始姿态；当前框架使用 yaw。",
    "agentScale": "初始缩放；本轮属性值异常为全 0，动态态势值为全 1。",
    "agentHp": "初始生命值。",
    "maxMoveSpeed": "最大速度字段，但本轮为 0，不能据此判断实体是否可移动。",
    "weaponCD": "武器冷却相关数值；士兵 0.2、机器狗 0.1，其余本轮为 0。",
    "bulletCnt": "初始弹药量。",
    "fireRange/guardRange/perceptionRange": "攻击、警戒、感知范围，三者不可混用。",
    "dodgeProb/explodeDmg": "闪避概率与爆炸伤害配置。",
    "debugAgent/isTeamReward": "调试与奖励模式标志。",
    "rSVD1": "propertyArr 中本轮为空；动态 dataArr 中用于‘实体名称;通信链路状态’。",
    "weaponType/color/rSVD2": "当前为空或未使用的预留字段。",
}

PROPERTY_TYPE_SUMMARY = {
    "BaseCommander_C": {"count": 2, "bulletCnt": 0, "weaponCD": 0, "fireRange": 1000,
                         "guardRange": 1400, "perceptionRange": 0},
    "BP_BaseSoldier_C": {"count": 30, "bulletCnt": 250, "weaponCD": 0.2,
                          "fireRange": 1000, "guardRange": 1400, "perceptionRange": 10000},
    "BP_Base_UAV_C": {"count": 10, "bulletCnt": 1, "weaponCD": 0, "fireRange": 1000,
                       "guardRange": 1400, "perceptionRange": 15000},
    "BP_RoboDog_C": {"count": 27, "bulletCnt": 250, "weaponCD": 0.1,
                      "fireRange": 1000, "guardRange": 1400, "perceptionRange": 8000},
    "BP_MNWS_Vehicle_6x6UGV_C": {"count": 3, "bulletCnt": 40, "weaponCD": 0,
                                  "fireRange": 1000, "guardRange": 1400,
                                  "perceptionRange": 0},
}


# ============================================================================
# 四、UE -> Python：reset_obs / step 响应
# ============================================================================

# 两种响应结构相同。reset_obs 本轮返回 70 条 dataArr，不含 2 名指挥员。
STEP_RESPONSE = {
    "valid": True,
    "dataArr": [{
        "valid": True,
        "agentAlive": True,
        "agentTeam": 0,
        "indexInTeam": 201,                   # 当前指挥员 UID
        "uId": 344,
        "maxMoveSpeed": 0,
        "agentLocation": {"x": 33422.1, "y": 22783.61, "z": 536.0455},
        "agentRotation": {"pitch": -0.007690107, "yaw": 148.965, "roll": -0.01462138},
        "agentScale": {"x": 1, "y": 1, "z": 1},
        "agentVelocity": {"x": 0, "y": 0, "z": 0},
        "agentHp": 100,
        "weaponCD": 0,
        "previousAction": -1,                 # 存活实体常见 -1；阵亡实体常见空串
        "availActions": [],                   # 见第五节感知结论
        "agentPerception": [],                # 本实体逐帧感知 UID
        "reward": 0,
        "isTeamReward": True,                 # 与属性回包 false 不一致
        "interaction": ["40.0"],             # 实测数值与弹药量对应，动态弹药量
        "type": "BP_MNWS_Vehicle_6x6UGV_C",
        "rSVD1": "”山猫”无人车_0;1",         # 名称;与当前指挥员通信是否正常，1代表正常，0代表中断
    }],
    "dataGlobal": {
        "valid": True,
        "teamReward": 0,
        "useTeamReward": False,
        "events": [],
        "visibleMatFlatten": [],              # 本轮始终为空
        "disMatFlatten": [],                  # 本轮始终为空
        "maxEpisodeStep": 200,
        "timeCnt": -1,                        # reset_obs=-1；step 回包依次 0..199
        "time": 0,
        "episodeDone": False,
        "episodeEndReason": "",
        "teamWin": -1,
        "keyObjArr": [{
            "valid": True,
            "uId": 32768,
            "className": "KeyObjDecoration_C",
            "location": {"x": 19235.14, "y": 33011.71, "z": 1152},
            "rotation": {"pitch": 0, "yaw": 34.50006, "roll": 0},
            "scale": {"x": 1, "y": 1, "z": 1},
            "velocity": {"x": 0, "y": 0, "z": 0},
            "hp": -1,                         # 未提供/不适用，不能当作实际生命
            "rSVD1": "",
        }],
        "levelName": "Matchlvl",
        "distanceMat": {
            "flat_arr": [0.0, 24875.56, 24928.63, ...],
            "shape": [70, 70],
        },
        "rSVD1": "1;1",                      # 总回合数=1，训练开关=1
    },
}

DATA_ITEM_FIELDS = {
    "valid": "该实体动态数据是否有效。",
    "agentAlive/agentHp": "当前存活与生命值；阵亡实体仍保留固定槽位。",
    "agentTeam/uId/type": "阵营、动态 UID、技术类型。",
    "indexInTeam": (
        "当前指挥员 UID。它描述单位的指挥隶属/通信对象，可动态变化；指挥员也可能是士兵。"
        "框架内部另建 team_index 仅用于稳定动作数组顺序，两者含义完全不同。"
    ),
    "agentLocation/agentVelocity": "当前位置与速度，三维坐标。",
    "agentRotation/agentScale": "当前姿态与缩放。",
    "maxMoveSpeed/weaponCD": "速度与冷却相关字段；本轮部分数值不可靠。",
    "previousAction": "上一动作字段；本轮通常为 -1，不能据此确认执行效果。",
    "agentPerception": "本实体感知列表。同队同帧内容不同，不是队伍统一列表。",
    "availActions": (
        "UE 返回的队伍级候选敌方 UID 池，不是动作名称、离散动作编号或单体动作掩码。"
        "本轮同队存活实体在全部状态帧中共享同一列表，阵亡实体通常为空。"
    ),
    "interaction": "动态弹药量",
    "reward/isTeamReward": "UE 奖励字段；当前训练奖励主要由 Python 算法计算。",
    "rSVD1": (
        "实体名称与通信链路状态，格式为‘名称;状态’。状态 1 表示当前单位与其指挥员"
        "通信链路正常，0 表示链路中断。本轮红方 8 架无人机在第 94～95 帧由 1 变为 0。"
    ),
}

GLOBAL_FIELDS = {
    "valid": "全局数据有效标志。",
    "teamReward/useTeamReward": "UE 队伍奖励值及是否启用。",
    "events": "本帧事件字符串列表，采用连续的 <字段>值 格式。",
    "maxEpisodeStep/timeCnt/time": "最大步数、离散帧计数、UE 仿真时间。",
    "episodeDone/episodeEndReason/teamWin": "结束标志、原因文本和胜方编号。",
    "keyObjArr": "关键目标数组，数量可变；本轮只有一个指挥所装饰对象。",
    "levelName": "UE 关卡名。",
    "distanceMat": "70×70 三维距离矩阵；flat_arr 按 UID 升序、行优先展开。",
    "visibleMatFlatten/disMatFlatten": "本轮为空的旧版/预留矩阵字段。",
    "rSVD1": "当前代码按‘总回合数;训练开关’解析，例如 1;1。",
}


# ============================================================================
# 五、感知字段的本轮实测结论
# ============================================================================

PERCEPTION_FINDINGS = {
    "reset_obs": "红蓝双方所有 agentPerception 与 availActions 均为空。",
    "first_nonempty": (
        "红方两字段均在 timeCnt=1 首次非空；蓝方 agentPerception 在 timeCnt=6、"
        "availActions 在 timeCnt=7 首次非空。"
    ),
    "agentPerception": (
        "单实体动态感知。同队同帧存在不同列表：timeCnt=10 时，蓝方士兵_12仅为 [319]，"
        "蓝方无人机_4 则包含 26 个红方 UID。因此它不是完整的队伍级共享列表。"
    ),
    "availActions": (
        "本轮可确定为队伍级共享候选敌方 UID 池：包含 reset_obs 在内的 201 个状态回包中，"
        "同队所有存活实体每帧均持有相同列表；阵亡实体通常为空。它不是动作类型列表，"
        "也不能解释为某个 Agent 的局部可攻击目标。"
    ),
    "team_union_relation": (
        "将 agentPerception 合并为全队并排除当帧已死亡敌方后，availActions 与该集合在红方 "
        "196/200 帧、蓝方 192/200 帧完全相等。少数差异集中在前期快速变化帧，说明二者高度"
        "相关，但存在更新时序差异，不能规定为严格恒等关系。"
    ),
    "range_relation": (
        "perceptionRange 与 fireRange 是两级范围。士兵本轮分别为 10000 cm 和 1000 cm；"
        "目标进入 agentPerception 只表示被感知，不表示已经进入武器射程。"
    ),
    "framework": (
        "发布框架用 agentPerception 构造单兵局部敌方对象和动态攻击候选；将 availActions "
        "与全队局部感知并集合并后，以 visible_opponents 脱敏队伍对象公开。框架不替选手按 "
        "fireRange、单位类型或通信状态二次过滤固定动作。"
    ),
}


# ============================================================================
# 六、事件、finish 与回合结束
# ============================================================================

EVENT_EXAMPLES = [
    "<Event>Destroyed<DamageCauser>111<Target>86",
    "<Event>EndEpisode<EndReason>RoundMaxCntReached<WinTeam>1",
]

EVENT_FIELDS = {
    "Event": "事件类型；本轮出现 Destroyed 和 EndEpisode。",
    "DamageCauser": "造成摧毁的实体 UID。",
    "Target": "被摧毁实体 UID。",
    "EndReason": "结束原因；本轮为达到最大回合步数。",
    "WinTeam": "胜方：0 红、1 蓝、-1 未定/需 Python 裁决。",
}

# UE 在最终 frame 之外还会主动发送 finish。它主要面向 UE 展示/计分；当前 Python
# 回合判定仍以最终 frame 的 dataGlobal 和 events 为准。本轮观察到重复 finish，框架忽略
# 无等待者的重复消息，不会据此重复结算。
FINISH_RESPONSE = {
    "Events": [{
        "Time": "19:04:39/100",
        "摧毁方实体": "士兵_12",
        "摧毁方": "Blue",
        "被摧毁方": "red",
        "被摧毁方实体": "士兵_7",
        "得分": "（蓝方+100分)",
    }],
    "Score": {},
    "Command": [],
    "Reward": 0,
}


# ============================================================================
# 七、Python 端解析规则（不是 UE 原始 JSON 的一部分）
# ============================================================================

PYTHON_PROCESSING = (
    "propertyArr 中只保留 team=0/1；BaseCommander_C 单独放入 commanders，不进入战斗动作列表。",
    "战斗实体按 reset_property 建立本回合内部清单；每帧 dataArr 按 uId 对齐，不能依赖 UE 原始数组顺序。",
    "选手动作按本方 state.agents 行对应实体；编码后的 StringActions 由末尾执行者 uId 标识单位，"
    "不要求复刻 propertyArr 的原始数组顺序。",
    "环境内部 Entity 保留完整 properties/raw；选手侧己方 raw 使用允许列表，敌方两者均为空。",
    "raw.indexInTeam 保留当前指挥员 UID；内部 team_index 只是动作数组下标，不能相互替代。",
    "raw.rSVD1 保留‘实体名称;通信链路状态’，可用于判断本实体与当前指挥员链路是否正常。",
    "agentPerception UID 只解析为本轮存活敌方；availActions 与全队局部感知并集合并后，"
    "映射为 visible_opponents 中的脱敏队伍级对象，不直接进入己方 raw。",
    "distanceMat 的索引顺序经坐标反算确认是 uId 数值升序，不是 dataArr 返回顺序。",
    "keyObjArr 全量保留，不能假设只有一个指挥所，也不能写死 UID 32768。",
    "事件字符串按 <字段>值 解析成字典；最终胜方优先读取 EndEpisode.WinTeam。",
)


# ============================================================================
# 八、2026-09-07 本轮一致性证据与边界
# ============================================================================

ROUND_20260907_EVIDENCE = {
    "algorithms": (
        "team=0 加载 T6fMRCbiXGZYXIvc，team=1 加载 QzGFknOfClsr6vCR；双方均为 rule。"
    ),
    "actions": (
        "200 个 step × 70 个战斗实体 = 14,000 条 StringActions；全部匹配 "
        "ActionSet2::<动作>;...;<执行者UID> 的基本结构。"
    ),
    "events": "10 条 Destroyed 与最终总死亡数 10 一致；最终包含一条 EndEpisode。",
    "communication": (
        "红方 8 架无人机伴随 UE 异常持续升高，在 timeCnt=94～95 从通信状态 1 切换为 0；"
        "这证明链路状态是逐帧动态字段。"
    ),
    "ue_boundary": (
        "UE 日志没有动作字符串拒绝记录，但仍存在无人机异常升高、无人车不移动、"
        "Replicated TSet/TMap、可视化数组越界等 UE 侧问题。‘未报解析错’只能证明字符串"
        "被接收，不能单独证明动作产生预期战术效果。"
    ),
}
