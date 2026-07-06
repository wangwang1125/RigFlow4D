# 图片/视频统一、4D、相机可选、骨架平级的人体动作捕捉方案清单

## 0. 项目一句话定义

> 输入人体图片或视频、任意一个人体目标骨架，以及可选的多视角/相机几何提示，直接预测该目标骨架的逐帧局部旋转。不经过 SMPL 或其他固定标准骨架的二次重定向，也不把相机内外参作为硬性输入前提。

整体目标：

\[
\{\text{Image/Video Observations},\ \text{Target Human Rig},\ \text{Optional Camera Hints}\}
\rightarrow
\{P^{S}_{1:T_q},R^{S}_{1:T_q},U^{S}_{1:T_q}\}
\]

其中：

- \(S\) 可以是 SMPL、Mixamo、UE Mannequin、MetaHuman 或自定义人体 Rig；
- \(P^S\) 是目标骨架拓扑下的 3D 关节位置；
- \(R^S\) 是目标骨架自身局部坐标系下的旋转；
- \(U^S\) 是接触、速度、不确定度和多解 motion latent；
- 输入可以是单张图片、多视角图片、单目视频或多视角视频；单张图片视为 \(T=1\)，视频视为 \(T>1\)；
- 所有骨架处于同一层级，SMPL 不是中间表示。
- 相机内参、外参、射线或视角 ID 是可用时的几何锚点，不可用时由模型估计相机关系或学习隐式跨视角对应。

---

## 一、从 MoCapAnything V2 得到的基础结论

- [ ] Video2Pose 根据目标骨架的关节集合构造 joint queries。
- [ ] 目标骨架有 \(J\) 个关节，就预测 \(T\times J\times3\) 的关节位置。
- [ ] 不同骨架通过 padding 和 joint mask 放入同一个模型。
- [ ] 显式 3D Pose 是动作内容与骨架旋转参数之间的有效中间表示。
- [ ] Rest Pose 只能描述骨架结构，不能完整定义局部旋转坐标轴。
- [ ] 目标 Rig 需要至少一个 pose–rotation reference pair 来校准局部坐标系。
- [ ] Pose2Rot 必须可学习，才能让旋转损失反向指导前面的姿态预测。
- [ ] 仅使用 3D 位置无法唯一恢复 bone-axis twist。
- [ ] 不必使用 Mesh 作为中间表示，可以直接从视觉特征预测目标骨架位置。
- [ ] 当前工程的图像特征主要来自 DINOv2 预提取 `image_embed`；新方案应升级为 DINOv3-first 的视觉 backbone，并保留可插拔接口。

这些设计构成项目起点，但研究范围限制为人体，并扩展到多视角和骨架平级解码。

---

## 二、需要避免的旧管线

不采用：

```text
图片/视频观测
    ↓
SMPL 参数或 SMPL 骨架
    ↓
传统 Retargeting / IK
    ↓
目标骨架旋转
```

原因：

- [ ] SMPL 到目标骨架仍存在层级差异；
- [ ] 脊柱、肩部、手臂关节数量可能不同；
- [ ] 局部坐标轴定义不同；
- [ ] twist bone 数量与位置不同；
- [ ] SMPL 的旋转误差会传递给目标骨架；
- [ ] 最终旋转损失无法直接优化多视角视觉模块；
- [ ] SMPL 会成为信息瓶颈。
- [ ] 不应把相机标定也变成新的硬瓶颈；已知相机参数应提升精度，但缺失标定时系统仍应能退化为自标定或隐式跨视角融合。

---

## 三、推荐的整体架构

### 3.1 总体流程

```text
人体图片或视频输入
单图 / 多视角图片 / 单目视频 / 多视角视频
        ↓
相机可选的视角-时间关系编码器
已知相机 / 弱相机提示 / 无相机参数
        ↓
4D Rig-native 人体运动场
F(t, chain, u)
        ↓
目标骨架 Joint Queries
        ↓
目标骨架位置与旋转联合解码
        ↓
可微 FK + 多视角重投影
        ↓
迭代修正位置与旋转
```

数学形式：

\[
M=E_{\mathrm{mv}}
\left(
I^{1:V}_{1:T_o},\tilde C^{1:V},\mathcal T_q
\right)
\]

\[
(\hat P^S,\hat R^S)
=
D
\left(
M,Q(S,A^S)
\right)
\]

其中：

- \(M\)：多视角共享的人体运动特征；
- \(T_o\)：观测帧数，图片为 1，视频大于 1；
- \(\mathcal T_q\)：需要输出动作的时间查询，可以等于观测帧，也可以是单图条件下的多个候选 motion queries；
- \(\tilde C\)：可选相机提示，可以是真实内外参、加噪参数、视角 ID、EXIF、估计相机或空值；
- \(S\)：目标骨架定义；
- \(A^S\)：目标骨架旋转坐标系的参考 anchor；
- \(Q\)：目标骨架的关节查询；
- \(D\)：骨架条件解码器。

相机参数不作为必需输入，而是作为可用时的强先验。模型同时预测或维护一个跨视角关系状态：

\[
G=E_{\mathrm{cam}}\left(I^{1:V}_{1:T_o},\tilde C^{1:V}\right)
\]

其中 \(G\) 可以包含相对相机位姿、基础矩阵/极线关系、跨视角 token 对应、深度概率或不确定度。已知相机时 \(G\) 被真实几何约束；未知相机时 \(G\) 由图像与人体运动一致性自估计。

---

## 四、目标骨架需要提供的信息

每个目标 Rig 至少准备：

```python
target_rig = {
    "parents":              [J],
    "rest_offsets":         [J, 3],
    "joint_names":          [J],
    "joint_mask":           [J],

    "chain_ids":            [J],
    "chain_coordinates":    [J],

    "reference_positions":  [K, J, 3],
    "reference_rotations":  [K, J, 6],

    "joint_limits":         optional,
    "twist_flags":          optional,
}
```

具体包括：

- [ ] 父子层级 `parents`
- [ ] Rest Pose 相对父节点偏移
- [ ] 关节名称和左右语义
- [ ] 有效关节 mask
- [ ] 所属人体运动链
- [ ] 在运动链上的归一化位置 \(u\)
- [ ] 一个或多个参考姿态
- [ ] 参考姿态对应的真实局部旋转
- [ ] 可选的关节限位
- [ ] 可选的主关节、twist bone 标记

---

## 五、骨架平级设计

### 核心原则

SMPL、Mixamo、UE、自定义 Rig 都作为条件输入：

\[
S\in
\{
S_{\text{SMPL}},
S_{\text{Mixamo}},
S_{\text{UE}},
S_{\text{Custom}}
\}
\]

共享同一个相机可选的多视角编码器：

```python
motion_feature, view_geometry = multiview_encoder(
    images,
    camera_hints=None,  # known / noisy / partial / missing
)

smpl_output = rig_decoder(motion_feature, smpl_rig)
ue_output = rig_decoder(motion_feature, ue_rig)
custom_output = rig_decoder(motion_feature, custom_rig)
```

- [ ] 不把 SMPL 当作教师骨架。
- [ ] 不要求目标骨架先与 SMPL 一一对应。
- [ ] 不在推理阶段执行 SMPL-to-Rig retargeting。
- [ ] 各骨架直接从共享人体运动特征读取动作。
- [ ] 每个骨架输出自己的局部旋转。
- [ ] 相机参数缺失时不切换到另一条管线，而是在同一模型内使用自标定/隐式视角关系分支。

---

## 六、骨架无关的人体运动表示

### 6.1 不推荐固定 24 关节作为唯一共享表示

即使不叫 SMPL，固定 24 关节也可能成为“隐形标准骨架”。

固定关节表示难以处理：

- 3 个与 6 个脊柱关节；
- 不同 clavicle 层级；
- forearm twist 数量不同；
- 手腕和手掌辅助骨不同；
- 脖子和头部层级不同。

### 6.2 推荐连续人体运动链

定义若干语义链：

- [ ] pelvis/root
- [ ] torso
- [ ] neck-head
- [ ] left arm
- [ ] right arm
- [ ] left leg
- [ ] right leg
- [ ] 可选 left/right hand
- [ ] 可选 finger chains

每条链使用连续坐标：

\[
u\in[0,1]
\]

模型学习：

\[
F_{t,c}(u)
\]

其中：

- \(c\) 是人体链；
- \(u\) 是链上的连续位置；
- \(F\) 表示该位置的时序运动特征。

目标关节 \(j\) 的链坐标可以由 Rest Pose 路径长度得到：

\[
u_j=
\frac{
\text{root of chain 到 joint }j\text{ 的路径长度}
}{
\text{该语义链的总长度}
}
\]

目标关节特征：

\[
h_{t,j}=F_{t,c_j}(u_j)
\]

这样，3 个脊柱关节和 6 个脊柱关节都从同一连续躯干场取样。

### 6.3 4D Rig-native Motion Field

将连续人体运动链扩展到时间维度：

\[
\mathcal F(t,c,u)
=
\{
x(t,c,u),
v(t,c,u),
swing(t,c,u),
\tau(t,c,u),
contact(t,c,u),
\sigma(t,c,u)
\}
\]

其中：

- \(x\)：链上连续位置；
- \(v\)：速度；
- \(swing\)：由骨段方向约束的摆动分量；
- \(\tau\)：累计 twist；
- \(contact\)：脚、手等端点接触状态；
- \(\sigma\)：不确定度或多解置信度。

目标 Rig 的每个关节只是对 \(\mathcal F(t,c,u)\) 的采样：

\[
h_{t,j}=\mathcal F(t,c_j,u_j)
\]

图片输入时 \(t\) 只有观测时刻，模型只应确定当前姿态，并可输出潜在运动分布。视频输入时，模型使用时间上下文恢复完整 4D 动作轨迹。多视角视频则同时利用视角几何和时间连续性，是最高精度模式。

---

## 七、目标关节 Query 设计

每个关节建立：

\[
q_j=
\phi
\left(
o_j,\pi_j,c_j,u_j,e_j,A_j
\right)
\]

包括：

- [ ] Rest Pose offset \(o_j\)
- [ ] 父节点与祖先关系 \(\pi_j\)
- [ ] 所属运动链 \(c_j\)
- [ ] 链上位置 \(u_j\)
- [ ] 关节名称语义 \(e_j\)
- [ ] 参考旋转 anchor \(A_j\)
- [ ] 是否为 twist bone
- [ ] 是否为末端关节
- [ ] 左右侧标签
- [ ] 可选自由度或关节限位

每个 query 要完成两件事：

1. 从共享人体运动场中获取动作；
2. 从图片/视频的视角-时间 token 中寻找与该目标关节相关的视觉证据。

---

## 八、相机可选的多视角编码部分

### 8.1 DINOv3-first 视觉特征提取

新方案将视觉特征提取从 DINOv2 升级为 DINOv3：

```text
image / frame
    ↓
human crop / mask / optional background removal
    ↓
DINOv3 dense visual tokens
    ↓
multi-view temporal tokens
    ↓
camera-optional relation encoder
```

设计原则：

- [ ] 默认使用 DINOv3 dense patch tokens，而不是只用 CLS token；
- [ ] 保留 register tokens / global tokens，用于全身朝向、遮挡和跨视角关系估计；
- [ ] 对手腕、前臂、肩、脊柱等 twist 关键区域保留更高分辨率局部 token；
- [ ] DINOv3 backbone 可以冻结、LoRA 微调或末端少量层微调；
- [ ] 视觉 token 需要通过 `visual_proj` 投影到统一 `q_dim`，避免模型主体被某一个 DINOv3 变体的维度绑定；
- [ ] 预处理缓存不再只记录 `image_embed`，还应记录 `visual_backbone_name`、`token_layout`、`patch_size`、`feature_dim` 和是否包含 register tokens。

推荐第一版使用：

```python
visual_backbone = {
    "family": "DINOv3",
    "variant": "ViT-L/16 or stronger",
    "tokens": "cls + registers + dense patches",
    "feature_dim": "backbone-dependent",
    "freeze": True,
    "projection_dim": q_dim,
}
```

DINOv3 是主干选择，不是核心论文贡献本身。核心贡献仍是骨架平级解码、连续 Rig-native motion field、相机可选旋转感知融合。DINOv3 的作用是提供更强、更稳定的 dense visual tokens，尤其服务于遮挡、跨视角对应和 twist 观测。

### 8.2 图片/视频统一的 4D 输入接口

输入统一表示为：

\[
I\in
\mathbb R^{B\times V\times T_o\times H\times W\times3}
\]

其中 \(V\) 是视角数，\(T_o\) 是观测时间长度。四种输入都使用同一套张量协议：

```text
单张图片:       V=1, T_o=1
多视角图片:     V>1, T_o=1
单目视频:       V=1, T_o>1
多视角视频:     V>1, T_o>1
```

模型输出由时间查询 \(\mathcal T_q\) 控制：

- [ ] 图片输入默认只输出当前帧 pose/rotation/contact/uncertainty；
- [ ] 图片输入可以额外输出多个 plausible motion latent，但不能当作确定未来真值；
- [ ] 视频输入输出与观测帧对齐的 4D 动作序列；
- [ ] 多视角视频输出最完整的 4D Rig-native motion field；
- [ ] 缺帧或异步时使用 `frame_time` 和 `observed_time_mask`，不强制所有视角同帧。

缓存和 batch 字段建议：

```python
visual_observation = {
    "input_type":             "image | multiview_image | video | multiview_video",
    "images":                 [V, T_o, H, W, 3],
    "visual_tokens":          [V, T_o, P, D],
    "frame_times":            [V, T_o],
    "observed_time_mask":     [V, T_o],
    "target_time_queries":    [T_q],
}
```

核心思想：4D 指的是 \(t\)-conditioned 人体运动场，不是必须输出 4D mesh 或 4D Gaussian。4D mesh/4DGS 可以作为辅助监督、可视化或额外分支，但主输出仍是目标 Rig 的局部旋转和动画。

### 8.3 基础输入

\[
I\in
\mathbb R^{B\times V\times T_o\times H\times W\times3}
\]

必需输入：

- [ ] 人体图片/视频图像或视觉 token；
- [ ] 目标 Rig 定义；
- [ ] 视角数量与时间索引。

可选输入：

- [ ] 相机内参；
- [ ] 相机外参；
- [ ] 相机中心；
- [ ] 像素射线方向；
- [ ] 视角 ID；
- [ ] 粗略相机位姿、EXIF 或焦距范围；
- [ ] 可选时间偏移；
- [ ] 2D 关键点或热力图；
- [ ] 可见性；
- [ ] 遮挡置信度。

输入模式分为三档：

```text
Mode A: calibrated
    图像 + 准确 K/R/t
    用显式投影、射线、极线和重投影损失获得最高精度。

Mode B: weak-calibrated
    图像 + 粗略 K/R/t、视角 ID、焦距范围或相机排序
    模型修正相机关系，并对相机噪声保持鲁棒。

Mode C: uncalibrated
    只有多视角图像
    模型学习隐式跨视角对应，或预测相对相机状态，再服务于目标 Rig 解码。
```

### 8.4 几何感知融合

不要只做多视角特征平均。

可采用：

- [ ] Epipolar attention
- [ ] Ray-based attention
- [ ] 3D query projection
- [ ] 多视角三角化特征
- [ ] 可学习深度概率
- [ ] 图像特征与相机几何交替更新
- [ ] 当相机参数缺失时，使用 learned view relation tokens 替代显式几何

对每个目标关节：

\[
F_{t,j}
=
\sum_v
w_{t,j,v}F_{t,j,v}
\]

### 8.5 相机关系估计分支

为了不强制要求外部标定，增加一个 camera relation head：

\[
\hat G_{u,v,t}
=
H_{\mathrm{cam}}
\left(
F^u_t,F^v_t,\tilde C^u,\tilde C^v
\right)
\]

输出可以包括：

- [ ] 两视角相对旋转和平移方向；
- [ ] 基础矩阵或极线 token；
- [ ] 每个视角的深度尺度与不确定度；
- [ ] 关节级跨视角匹配置信度；
- [ ] 相机参数是否可信的 gate。

训练时真实相机参数只作为监督或增强来源，而不是推理时的硬依赖。模型应在 calibrated、weak-calibrated、uncalibrated 三种模式下共用主体网络。

---

## 九、相机可选、旋转感知的多视角选择

这是重要创新候选。

普通多视角方法选择“哪个视角最适合估计关节位置”，本项目选择：

> 哪个视角最适合估计目标骨架某个关节的某种旋转分量。

视角权重：

\[
w_{t,j,v}
=
\operatorname{softmax}_v
g(q_j,f_{t,j,v},\sigma_{t,j,v},\hat G_{t,v})
\]

其中：

- \(q_j\)：目标关节特征；
- \(f_{t,j,v}\)：该视角的局部图像特征；
- \(\sigma_{t,j,v}\)：该视角的不确定度。
- \(\hat G_{t,v}\)：该视角的相机几何状态或隐式视角关系。

进一步分为：

\[
w^{\text{swing}}_{t,j,v},
\qquad
w^{\text{twist}}_{t,j,v}
\]

- [ ] swing 主要依赖关节空间方向；
- [ ] twist 更多依赖手掌方向、衣服纹理、身体轮廓和时间变化；
- [ ] 不同自由度可以选择不同视角；
- [ ] 目标骨架不同，视角权重也可以不同。
- [ ] 已知相机时，视角权重利用射线、极线和投影残差；
- [ ] 未知相机时，视角权重利用跨视角 token 对应、相对人体朝向和时间一致性；
- [ ] 模型输出每个视角、每个关节、每个旋转分量的可观测性与置信度。

---

## 十、Twist Bone 专项设计

### 10.1 问题

不同人体骨架可能分别具有：

```text
Rig A:
upperarm → lowerarm → hand

Rig B:
upperarm
 → upperarm_twist_01
 → upperarm_twist_02
 → lowerarm
 → lowerarm_twist
 → hand
```

SMPL 的总扭转无法直接确定如何分配到 Rig B。

### 10.2 连续 Twist Field

对每条肢体预测累计 twist：

\[
\tau_{t,c}(u)
\]

关节局部 twist 由相邻位置差得到：

\[
\Delta\tau_{t,j}
=
\tau_{t,c}(u_j)
-
\tau_{t,c}(u_{\pi_j})
\]

旋转拆分：

\[
R_{t,j}
=
R^{\text{swing}}_{t,j}
R^{\text{twist}}_{t,j}
\]

检查项：

- [ ] 无 twist bone 的骨架可以集中分配 twist；
- [ ] 多 twist bone 的骨架可以连续分配；
- [ ] 分配依据 Rest Pose 中的链坐标；
- [ ] twist 由多视角外观和时间特征预测；
- [ ] 使用参考旋转 anchor 校准正负方向与局部轴；
- [ ] 加入 twist 的时间平滑损失；
- [ ] 对 wrist、forearm、upperarm、spine 单独评估。

---

## 十一、参考姿态与局部坐标系校准

每个 Rig 使用：

\[
A^S=
\{
(P^{ref}_k,R^{ref}_k)
\}_{k=1}^{K}
\]

推荐参考动作：

- [ ] T-pose 或 A-pose；
- [ ] 肘部弯曲；
- [ ] 前臂旋转；
- [ ] 手掌翻转；
- [ ] 躯干扭转；
- [ ] 膝盖弯曲；
- [ ] 脚踝转动。

可以提出 **K-shot Rig Calibration**：

\[
C_{t,j}
=
\sum_{k=1}^{K}
\alpha_{t,j,k}C^{ref}_{j,k}
\]

不同关节、不同动作动态选择最有信息的参考姿态。

实验对比：

- [ ] Rest Pose only
- [ ] 单个 reference pair
- [ ] 随机多个 reference pairs
- [ ] 主动选择的信息最大参考姿态
- [ ] 不同 \(K\) 值：1、2、4、8

---

## 十二、位置与旋转联合预测

不必严格采用：

\[
Video\rightarrow P\rightarrow R
\]

可以采用联合迭代：

\[
(P^{k+1},R^{k+1})
=
D(M,S,P^k,R^k)
\]

流程：

```text
目标关节 Queries
      ↓
预测初始位置 P¹
      ↓
预测初始旋转 R¹
      ↓
FK(R¹) 得到位置 P_FK¹
      ↓
投影回可信相机，或与估计视角关系做一致性检查
      ↓
结合重投影残差更新 Queries
      ↓
预测 P² 和 R²
```

- [ ] 位置约束 swing；
- [ ] 图像外观补充 twist；
- [ ] 时间上下文消除旋转翻转；
- [ ] FK 保证旋转与位置一致；
- [ ] 已知相机时用多视角重投影检查输出是否解释所有相机；
- [ ] 未知相机时用跨视角特征一致性、相对人体朝向一致性和估计相机关系残差替代硬重投影；
- [ ] 旋转损失直接更新多视角特征融合。

### 12.2 Latent Flow Kinematic Refinement

不建议用 mesh VAE latent 直接作为骨架平移空间。更推荐在 4D Rig-native motion field 上训练一个运动潜空间：

```text
DINOv3 view-time tokens
        ↓
deterministic pose head → P0
        ↓
Kinematic VAE encoder
GT 4D field / chain samples / contact → z_kin
        ↓
Conditional flow matching
noise → z_kin
condition: visual tokens + rig query + camera relation + P0
        ↓
Kinematic VAE decoder
ΔP / refined P / contact / uncertainty / motion latent
        ↓
Pose2Rot + FK consistency
```

核心约束：

- [ ] VAE latent 是 rig-native kinematic latent，不是 SMPL latent，也不是 mesh latent；
- [ ] flow matching 负责修正、补全和表达多解性，不负责从零替代确定性 pose head；
- [ ] 干净多视角视频使用 \(P_0\) 保持精度；
- [ ] 图片、少视角、遮挡、无标定输入使用 latent flow 生成合理 motion hypothesis；
- [ ] decoder 输出 \(\Delta P\)、contact 和 uncertainty，供 Pose2Rot 和 FK 闭环使用。

训练目标：

\[
z_{\mathrm{kin}}
=
E_{\mathrm{kin}}
\left(
\mathcal F^{GT}
\right)
\]

\[
v_\theta(z_t,t\,|\,M,S,\hat G,P_0)
\approx
z_{\mathrm{kin}}-\epsilon
\]

推理时从噪声或先验 latent 经 flow 得到 \(\hat z_{\mathrm{kin}}\)，再解码为 4D motion field 的修正项。该模块作为高风险场景增强贡献，若实验提升明显，可以升为核心贡献。

---

## 十三、训练数据清单

### 13.0 公开数据集组合建议

本项目不建议押注单一数据集。我们的目标同时包含图片/视频统一、多视角可选相机、4D motion field、目标 Rig 原生旋转和 latent flow refinement，因此训练数据应按功能分层：

```text
AMASS / AIST++                 → 运动潜空间、VAE、flow、rotation prior
BEDLAM                         → 合成视觉预训练、遮挡、衣服、场景与相机运动
MVHumanNet++ / HuMMan          → 主视觉训练，多视角/视频/人体 4D 标注
3DPW / EMDB / RICH / AIST++    → 泛化评测，野外、移动相机、接触和长时序动作
```

推荐优先级如下：

| 层级 | 数据集 | 主要价值 | 在本项目中的用法 |
| --- | --- | --- | --- |
| 主训练 | **MVHumanNet++** | 大规模多视角人体动作序列，包含 camera params、2D/3D keypoints、SMPL/SMPLX、mask、normal、depth 等标注 | 作为 `DINOv3 + View-Time Relation Transformer + Skeleton-Peer Decoder` 的主训练数据 |
| 主训练备选/补充 | **HuMMan** | 多模态 4D 人体数据，含 color images、point clouds、keypoints、SMPL parameters、textured meshes | 当 MVHumanNet++ 获取、清洗或算力成本过高时，作为主训练替代；也可补充跨设备域差异 |
| 运动先验 | **AMASS** | 大规模 mocap motion 参数集合，无图像但动作覆盖广 | 训练 kinematic VAE、latent flow、contact/velocity/rotation prior；不单独用于视觉监督 |
| 合成视觉桥接 | **BEDLAM** | 合成 RGB video + SMPL-X GT，含 realistic clothing、segmentation、depth、camera motion | 预训练视觉到人体运动映射，增强衣服、遮挡、体型、背景和相机运动泛化 |
| 大动作/长时序 | **AIST++** | 舞蹈视频与 3D motion，动作幅度大、节奏强、长时序明显 | 训练/评测 4D motion field、temporal smoothness、flow motion hypothesis |
| 野外单目评测 | **3DPW** | moving phone camera，带 2D/3D pose 和 camera poses | 评测单目视频、移动相机和无/弱相机提示模式 |
| 全局轨迹评测 | **EMDB** | in-the-wild videos，含 SMPL pose/shape、global body trajectory、global camera trajectory | 评测 root motion、global motion、camera-optional relation head |
| 接触评测 | **RICH** | 多视角 4K 室内外人体-场景交互，含 dense body-scene contact labels | 评测 contact head、foot/hand contact stability 和 FK 接触一致性 |

第一版实现建议采用：

```text
Stage A: AMASS + AIST++ motion-only
    训练 kinematic VAE、rotation/contact/velocity prior

Stage B: BEDLAM synthetic visual pretrain
    训练 DINOv3 token projection、单目/视频 pose seed、遮挡鲁棒性

Stage C: MVHumanNet++ 主训练
    训练多视角/视频融合、camera dropout、Skeleton-Peer Decoder、4D field

Stage D: HuMMan 补充或替代
    加强多模态/跨设备/多动作泛化

Stage E: 3DPW + EMDB + RICH + AIST++
    只做泛化评测或少量 validation，不作为核心训练泄漏测试集
```

注意事项：

- [ ] 不把 SMPL/SMPL-X 当成推理阶段的中间骨架，只把它们作为公开数据集的监督来源；
- [ ] 所有 SMPL/SMPL-X 标签都需要转换成统一 `positions/local_rotations/root_translation/contact` 字段，再由目标 Rig adapter 生成 rig-native label；
- [ ] 多视角数据中真实相机参数只作为训练监督和增强来源，推理时仍支持准确相机、加噪相机、仅视角 ID、无相机四种模式；
- [ ] 对 `3DPW`、`EMDB`、`RICH` 这类评测集应避免混入训练，防止野外泛化指标虚高；
- [ ] 数据集下载和许可证多需要注册，工程里只保存转换脚本、字段 schema 和 split 文件，不重新分发原始数据。

公开入口记录：

- MVHumanNet++: <https://kevinlee09.github.io/research/MVHumanNet%2B%2B/>
- HuMMan: <https://caizhongang.github.io/projects/HuMMan/>
- AMASS: <https://amass.is.tue.mpg.de/>
- BEDLAM: <https://bedlam.is.tue.mpg.de>
- AIST++: <https://google.github.io/aichoreographer/>
- 3DPW: <https://virtualhumans.mpi-inf.mpg.de/3DPW/>
- EMDB: <https://eth-ait.github.io/emdb/>
- RICH: <https://rich.is.tue.mpg.de/>

### 13.1 Rig-native 数据

每个骨架使用自己的原生动画：

```text
Rig A 动画 + 多视角渲染 → A 的真实旋转
Rig B 动画 + 多视角渲染 → B 的真实旋转
Rig C 动画 + 多视角渲染 → C 的真实旋转
```

每个样本需要：

```python
sample = {
    "input_type":             "image | multiview_image | video | multiview_video",
    "multiview_images":       [V, T, H, W, 3],
    "visual_tokens":          optional [V, T, P, D],
    "visual_backbone_name":   optional,
    "visual_token_layout":    optional,
    "frame_times":            optional [V, T],
    "observed_time_mask":     optional [V, T],
    "target_time_queries":    optional [Tq],
    "camera_intrinsics":      optional [V, ...],
    "camera_extrinsics":      optional [V, ...],
    "camera_valid_mask":      optional [V],
    "camera_noise_level":     optional,
    "view_ids":               optional [V],

    "parents":                [J],
    "rest_offsets":           [J, 3],
    "joint_names":            [J],
    "chain_ids":              [J],
    "chain_coordinates":      [J],

    "positions":              [T, J, 3],
    "local_rotations_6d":     [T, J, 6],
    "root_translation":       [T, 3],
    "velocities":             optional [T, J, 3],
    "chain_field_samples":    optional [T, C, U, Df],
    "kinematic_latent":       optional [Tz, Dz],

    "reference_positions":    [K, J, 3],
    "reference_rotations":    [K, J, 6],

    "contact_labels":         optional,
}
```

### 13.2 同动作、多 Rig 数据

为了学习骨架平级一致性，准备部分共享动作：

```text
同一个动作观测
   ├─ Rig A 的真实解
   ├─ Rig B 的真实解
   └─ Rig C 的真实解
```

动作来源尽量骨架中立：

- [ ] 光学 Mocap markers；
- [ ] IMU segment orientation；
- [ ] 多视角人体表面 landmark；
- [ ] 密集人体表面轨迹；
- [ ] DCC 原创动画约束；
- [ ] 真实多视角拍摄加人工校正。

避免把 SMPL 动画简单重定向到所有 Rig 后直接当成绝对真值，否则训练标签仍然被 SMPL 支配。

### 13.3 局部坐标轴增强

对同一 Rig：

- [ ] 改变关节局部坐标轴；
- [ ] 保持世界空间动画不变；
- [ ] 重新烘焙局部旋转；
- [ ] 重新生成 reference pair；
- [ ] 检查 FK 后世界位置一致。

目标是迫使模型读取 anchor，而不是记住某种统一轴定义。

### 13.4 多视角增强

训练时随机：

- [ ] 选择 1 到 \(N\) 个视角；
- [ ] 删除部分相机；
- [ ] 模拟相机掉线；
- [ ] 随机隐藏全部或部分相机内外参；
- [ ] 将准确相机参数替换为粗略参数、加噪参数或仅视角 ID；
- [ ] 遮挡手、脚、躯干；
- [ ] 添加运动模糊；
- [ ] 降低图像分辨率；
- [ ] 添加亮度和背景变化；
- [ ] 添加相机外参噪声；
- [ ] 添加轻微时间不同步；
- [ ] 随机改变相机布置；
- [ ] 单独遮挡能够观察 twist 的关键视角。

核心训练策略是 camera dropout：

```text
p=0.4: 使用准确相机参数
p=0.3: 使用加噪或不完整相机参数
p=0.3: 完全不提供相机参数
```

这样模型不会把相机标定记成必需条件，而是学习在不同几何可用性下保持局部旋转稳定。

### 13.5 DINOv3 特征缓存与增强

当前工程的缓存格式以 `image_embed` 为主，适配 DINOv2 的 `[T, P, 1024]` 视觉 token。DINOv3 迁移时需要把缓存升级为 backbone-aware 格式：

```python
visual_cache = {
    "visual_tokens":          [T, P, D],
    "visual_backbone_name":   "dinov3_vitl16",
    "visual_feature_dim":     D,
    "visual_patch_size":      16,
    "visual_token_layout":    {
        "has_cls": True,
        "num_registers": optional,
        "patch_grid": [H_p, W_p],
    },
}
```

训练增强：

- [ ] DINOv3 token dropout；
- [ ] patch token masking；
- [ ] view-token dropout；
- [ ] 高分辨率局部 crop token 与全身 token 混合；
- [ ] DINOv2/DINOv3 特征蒸馏或过渡训练；
- [ ] 固定 backbone 与轻量微调 backbone 的对比。

---

## 十四、损失函数清单

总损失可以写成：

\[
\mathcal L=
\lambda_{pos}\mathcal L_{pos}
+\lambda_{rot}\mathcal L_{rot}
+\lambda_{vel}\mathcal L_{rotVel}
+\lambda_{fk}\mathcal L_{FK}
+\lambda_{proj}\mathcal L_{reproj}
+\lambda_{chain}\mathcal L_{chain}
+\lambda_{contact}\mathcal L_{contact}
+\lambda_{view}\mathcal L_{view}
+\lambda_{anchor}\mathcal L_{anchor}
+\lambda_{cam}\mathcal L_{cam}
+\lambda_{4d}\mathcal L_{4d}
+\lambda_{flow}\mathcal L_{flow}
\]

### 必选损失

- [ ] 3D 关节位置误差
- [ ] 局部旋转 geodesic error
- [ ] 旋转角速度误差
- [ ] FK 位置一致性
- [ ] 有相机参数时的多视角 2D 重投影误差
- [ ] 无相机参数时的跨视角特征/关键点一致性误差
- [ ] 根节点朝向和位移误差
- [ ] 4D motion field 的位置、速度、contact 一致性

### 推荐损失

- [ ] reference frame 旋转一致性
- [ ] swing–twist 分解损失
- [ ] twist 累计场平滑损失
- [ ] 跨 Rig 语义端点一致性
- [ ] 连续链采样点一致性
- [ ] 足部接触损失
- [ ] foot sliding 损失
- [ ] 地面穿透损失
- [ ] 关节限位损失
- [ ] 任意视角子集输出一致性
- [ ] 不确定度校准损失
- [ ] 相机关系估计损失或相机置信度校准损失
- [ ] kinematic VAE 重建损失与 KL / VQ commitment 损失
- [ ] flow matching velocity 损失
- [ ] 图片输入下的多 hypothesis 多样性与置信度排序损失

---

## 十五、跨骨架一致性方式

不同 Rig 不直接逐关节比较，而是在语义空间比较。

### 15.1 语义端点

比较：

- pelvis
- head
- shoulders
- elbows
- wrists
- knees
- ankles
- toes

\[
\mathcal L_{\text{semantic}}
=
\sum_k
\left\|
X^A_{t,k}-X^B_{t,k}
\right\|_1
\]

### 15.2 连续运动链

在每条链采样：

\[
u_m\in\{0,0.1,\ldots,1\}
\]

比较不同 Rig 在相同 \(u_m\) 处的运动：

\[
\mathcal L_{\text{chain}}
=
\sum_{c,m}
\left\|
X^A_{t,c}(u_m)
-
X^B_{t,c}(u_m)
\right\|
\]

### 15.3 接触与速度

- [ ] 末端速度一致；
- [ ] 脚部接触状态一致；
- [ ] 手部接触状态一致；
- [ ] 运动节奏一致；
- [ ] 根节点运动方向一致。

---

## 十六、推荐训练阶段

### 阶段 0：坐标系与数据验证

- [ ] 验证 \(FK(R^{GT})\approx P^{GT}\)；
- [ ] 验证不同 Rig 的 local/global rotation 定义；
- [ ] 验证 6D rotation 转换；
- [ ] 验证 bind pose 与 rest pose；
- [ ] 验证相机投影；
- [ ] 验证相机参数隐藏、加噪和缺失时的数据字段；
- [ ] 验证 DINOv3 token 形状、token layout、feature_dim 与模型 `img_dim/visual_proj` 一致；
- [ ] 验证 DINOv2 旧缓存与 DINOv3 新缓存不能被静默混用；
- [ ] 验证图片/视频统一输入字段：`input_type`、`frame_times`、`observed_time_mask`、`target_time_queries`；
- [ ] 验证左右手坐标系。

### 阶段 0.5：公开数据集适配

先不训练主模型，只做数据统一：

- [ ] AMASS / AIST++：转换为 motion-only `positions/local_rotations/root_translation/contact/velocity`；
- [ ] BEDLAM：转换为 synthetic visual-video + SMPL-X label + camera/mask/depth optional fields；
- [ ] MVHumanNet++ / HuMMan：转换为主训练 `multiview_images + camera hints + 3D pose + SMPL/SMPLX + optional depth/normal/mask`；
- [ ] 3DPW / EMDB / RICH：单独建立 evaluation split，默认不参与训练；
- [ ] 为所有数据集生成统一 `sample` schema 和 `dataset_name/source_label_type/license_note`；
- [ ] 所有数据集 adapter 必须输出 `data.schema.RigFlowSample`，并在写入 cache 或进入 batch 前调用 `sample.validate()`；
- [ ] 建立 `rig_adapter`：SMPL/SMPL-X label 只作为监督来源，最终转成目标 Rig 的 rig-native supervision；
- [ ] 建立 camera dropout metadata：准确相机、加噪相机、仅视角 ID、完全无相机；
- [ ] 检查所有数据集的 frame rate、坐标轴、单位、root convention 和左右手定义。

### 阶段 1：单 Rig 的 Pose2Rot

先不使用视频：

\[
P^S,S,A^S\rightarrow R^S
\]

- [ ] 在一个 Rig 上过拟合；
- [ ] 检查 twist；
- [ ] 检查时间连续性；
- [ ] 检查 FK 还原；
- [ ] 检查 reference anchor 是否被使用。

### 阶段 2：多 Rig 平级 Pose2Rot

- [ ] 加入不同关节数；
- [ ] 加入不同脊柱层级；
- [ ] 加入不同 twist 配置；
- [ ] 加入随机局部轴；
- [ ] 使用 padding 与 mask；
- [ ] 测试完全未见 Rig。

### 阶段 3：多视角到目标 Pose

\[
\text{DINOv3 Multi-view Tokens}
+
S
\rightarrow
P^S
\]

- [ ] 先只监督位置；
- [ ] 先冻结 DINOv3，只训练视觉投影、多视角融合和目标骨架 decoder；
- [ ] 对比 DINOv2 image_embed 与 DINOv3 dense tokens；
- [ ] 验证目标关节 query；
- [ ] 验证可变关节数量；
- [ ] 验证多视角几何融合；
- [ ] 验证不同视角数量。
- [ ] 分别验证准确相机、加噪相机、无相机参数三种输入模式。
- [ ] 主训练数据优先使用 MVHumanNet++；若规模或清洗成本过高，则先用 HuMMan 建立可复现实验闭环。

### 阶段 3.1：图片/视频统一 4D Motion Field

\[
\text{Image/Video Tokens}
+
S
\rightarrow
\mathcal F(t,c,u)
\]

- [ ] 单图 \(V=1,T=1\)：只监督当前姿态与不确定度；
- [ ] 多视角图片 \(V>1,T=1\)：监督当前 3D pose、swing 和初始 contact；
- [ ] 单目视频 \(V=1,T>1\)：监督时间连续性、速度和 contact；
- [ ] 多视角视频 \(V>1,T>1\)：监督完整 4D motion field；
- [ ] 验证 `target_time_queries` 可以查询观测帧，也可以查询稀疏/滑窗时间点。
- [ ] 使用 BEDLAM 做合成视频预训练，使用 MVHumanNet++ / HuMMan 做真实多视角训练，使用 AIST++ 验证长时序大动作稳定性。

### 阶段 3.5：相机关系自估计

\[
\text{Multi-view}
+
\tilde C
\rightarrow
\hat G
\]

- [ ] 监督相对相机旋转、平移方向或极线关系；
- [ ] 在相机参数隐藏时预测跨视角对应；
- [ ] 用人体关键点、轮廓或 dense track 做跨视角一致性；
- [ ] 评估 \(\hat G\) 是否能提升无标定模式下的 pose/rotation。

### 阶段 4：端到端位置与旋转

\[
\text{Multi-view}
+
S
+
A^S
\rightarrow
(P^S,R^S)
\]

- [ ] 让旋转损失流回多视角编码器；
- [ ] 加入 FK 闭环；
- [ ] 加入有相机参数时的重投影闭环；
- [ ] 加入无相机参数时的跨视角一致性闭环；
- [ ] 加入 joint/DoF 级视角权重；
- [ ] 加入时间注意力。

### 阶段 5：连续运动链与 Twist Field

- [ ] 用连续链替代固定人体关节模板；
- [ ] 实现链坐标 \(u\)；
- [ ] 实现累计 twist field；
- [ ] 测试不同 twist bone 数量；
- [ ] 测试不同脊柱关节数；
- [ ] 比较固定 canonical skeleton 基线。

### 阶段 6：Latent Flow Kinematic Refinement

- [ ] 训练 kinematic VAE：\(\mathcal F^{GT}\rightarrow z_{\mathrm{kin}}\rightarrow \hat{\mathcal F}\)；
- [ ] 冻结或半冻结 VAE，训练条件 flow matching；
- [ ] 条件输入包含 DINOv3 tokens、目标 Rig queries、camera relation、初始 \(P_0\)；
- [ ] 先让 flow 预测 \(\Delta P\) 和 uncertainty，再扩展到 contact 和 latent motion hypothesis；
- [ ] 对比 deterministic head、latent flow refinement、纯 flow 生成三种模式。
- [ ] VAE 和 motion prior 优先使用 AMASS + AIST++ motion-only 训练，再用 MVHumanNet++ / HuMMan 的视觉条件做 conditional flow refinement。

---

## 十七、实验基线

至少比较：

- [ ] DINOv2 image_embed + 原始单视角/多视角融合
- [ ] DINOv3 frozen dense tokens + 相机可选融合
- [ ] DINOv3 lightly fine-tuned tokens + 相机可选融合
- [ ] 图片输入 \(T=1\) → 当前目标 Rig pose/rotation
- [ ] 视频输入 \(T>1\) → 4D Rig-native motion field
- [ ] 多视角图片 → 当前 3D pose + rotation
- [ ] 多视角视频 → 完整 4D motion + rotation
- [ ] 单目人体姿态 + IK
- [ ] 多视角 3D Pose + IK
- [ ] 多视角 → SMPL → 目标 Rig
- [ ] 多视角 → 固定 canonical skeleton → 目标 Rig
- [ ] 多视角 → 目标 Rig 直接回归旋转
- [ ] 多视角 → 目标 Pose → 可学习 Pose2Rot
- [ ] 骨架平级联合位置旋转解码
- [ ] 已知相机参数的骨架平级解码
- [ ] 加噪相机参数的骨架平级解码
- [ ] 无相机参数的骨架平级解码
- [ ] 无相机参数 + 自估计相机关系的骨架平级解码
- [ ] 固定离散人体骨架表示
- [ ] 连续人体运动链表示
- [ ] 无 reference anchor
- [ ] 单 reference anchor
- [ ] K-shot reference anchor
- [ ] 无 latent flow refinement
- [ ] latent flow refinement
- [ ] 纯 flow 生成，不使用 deterministic \(P_0\)

---

## 十八、重点消融实验

### 骨架表示

- [ ] SMPL 中间层
- [ ] 固定 24 关节中间层
- [ ] 无显式中间层
- [ ] 连续人体运动场

### 多视角融合

- [ ] 简单平均
- [ ] 全身共享 view weights
- [ ] 逐关节 view weights
- [ ] 逐关节、逐自由度 view weights
- [ ] 准确相机几何
- [ ] 加噪相机几何
- [ ] 仅视角 ID
- [ ] 无相机几何
- [ ] 有射线与极线信息
- [ ] 自估计相机关系
- [ ] 相机关系估计分支 detach / 不 detach

### 视觉 Backbone

- [ ] DINOv2
- [ ] DINOv3 frozen
- [ ] DINOv3 + LoRA / adapter
- [ ] DINOv3 global tokens only
- [ ] DINOv3 dense patch tokens
- [ ] DINOv3 dense patch tokens + register/global tokens
- [ ] 单尺度 token
- [ ] 全身 token + 局部高分辨率 token

### 图片/视频与 4D 表示

- [ ] 只支持视频
- [ ] 图片和视频统一接口
- [ ] 离散关节序列 \(P_{t,j}\)
- [ ] 连续 4D motion field \(\mathcal F(t,c,u)\)
- [ ] 无时间查询，仅输出观测帧
- [ ] 使用 `target_time_queries`
- [ ] 图片输入单 hypothesis
- [ ] 图片输入多 hypothesis + uncertainty

### Latent Flow

- [ ] 无 kinematic VAE
- [ ] kinematic VAE only
- [ ] deterministic head + flow refinement
- [ ] pure conditional flow generation
- [ ] flow 输出 \(\Delta P\)
- [ ] flow 输出 \(\Delta P + contact + uncertainty\)

### 旋转建模

- [ ] 直接回归完整旋转
- [ ] swing–twist 分解
- [ ] 无 twist field
- [ ] 连续 twist field

### Anchor

- [ ] Rest Pose only
- [ ] 单参考姿态
- [ ] 多参考姿态
- [ ] 随机参考姿态
- [ ] 主动选择参考姿态

### 训练方式

- [ ] 分阶段训练
- [ ] 旋转梯度 detach
- [ ] 完整端到端
- [ ] 无 FK loss
- [ ] 无 reprojection loss

---

## 十九、评价指标

### 位置

- [ ] MPJPE
- [ ] PA-MPJPE
- [ ] MPJVE
- [ ] 2D reprojection error
- [ ] FK position error

### 旋转

- [ ] local rotation geodesic error
- [ ] global rotation error
- [ ] angular velocity error
- [ ] twist angle error
- [ ] swing direction error
- [ ] root orientation error

### 动画质量

- [ ] foot sliding
- [ ] contact accuracy
- [ ] ground penetration
- [ ] joint limit violation
- [ ] jerk / acceleration
- [ ] 长序列漂移
- [ ] 4D motion field temporal consistency
- [ ] latent flow refinement gain
- [ ] 图片输入多 hypothesis 的多样性与最优样本误差

### 泛化能力

- [ ] 未见骨长比例
- [ ] 未见层级
- [ ] 未见关节数量
- [ ] 未见局部坐标轴
- [ ] 未见 twist bone 配置
- [ ] 未见相机布局
- [ ] 相机参数缺失
- [ ] 相机参数加噪
- [ ] 不同视角数量
- [ ] 严重遮挡
- [ ] 图片输入到视频输入的统一泛化
- [ ] 不同帧率和不规则时间采样

---

## 二十、第一版建议控制的范围

建议初版只做：

- [ ] 单人；
- [ ] 同步或轻微不同步相机；
- [ ] 固定或近固定相机；
- [ ] 训练数据保留真实相机参数用于监督和增强；
- [ ] 推理阶段相机参数可选，支持准确参数、加噪参数、仅视角 ID、无参数四种输入；
- [ ] 视觉特征使用 DINOv3 dense tokens，DINOv2 只作为消融和旧缓存兼容基线；
- [ ] 输入支持单张图片、多视角图片、单目视频、多视角视频；
- [ ] 图片输入只要求当前帧姿态与不确定度，不要求确定未来动作；
- [ ] 视频输入输出 4D Rig-native motion field；
- [ ] latent flow refinement 先作为可选增强分支；
- [ ] 人体 Body，不含面部；
- [ ] 手部先使用简化 wrist/palm，不做完整手指；
- [ ] 2 到 6 个视角；
- [ ] 多种人体 Rig；
- [ ] 不同脊柱和 twist bone 配置；
- [ ] 直接输出目标 Rig 局部旋转；
- [ ] 支持未见目标 Rig。

暂不加入：

- [ ] 多人；
- [ ] 完全无同步约束；
- [ ] 动态相机；
- [ ] 场景重建；
- [ ] 完整手指和面部；
- [ ] 衣物物理；
- [ ] 人与物体复杂交互。

---

## 二十一、推荐的核心创新

论文主线建议集中在前三根主梁上，第四、第五项作为增强贡献。若实验提升足够明显，可以将第四或第五项上升为核心贡献。

### 创新 1：Skeleton-Peer Direct Decoding

> SMPL 与其他人体 Rig 完全平级，目标旋转直接由共享多视角运动特征和目标骨架 queries 生成。

### 创新 2：Continuous Rig-Native Motion Field

> 使用连续的躯干、手臂和腿部运动链，同时预测位置、swing、累计 twist、速度和接触状态。目标 Rig 不是被重定向的结果，而是对连续人体运动场的原生采样。

### 创新 3：Camera-Optional Rotation-Aware Fusion

> 相机参数从硬性输入降级为可选几何锚点。目标关节按时间、关节和旋转自由度选择最有价值的视角；有标定时利用显式几何，无标定时通过自估计视角关系和隐式跨视角对应恢复局部旋转，尤其改善遮挡和 twist。

### 创新 4：Latent Flow Kinematic Refinement

> 在 rig-native 4D motion field 上训练 kinematic latent，并用条件 flow matching 修正、补全和表达多解性。Flow 不从零替代 pose head，而是在 \(P_0\) 的基础上输出 \(\Delta P\)、contact、uncertainty 和 motion hypothesis。

### 创新 5：Image/Video Unified 4D Motion Interface

> 图片、视频、多视角图片和多视角视频共享同一套 \(V\times T\) 输入协议。图片是 \(T=1\) 的 4D 查询，视频是 \(T>1\) 的 4D 观测；模型输出由 `target_time_queries` 控制，从当前帧姿态扩展到完整 Rig-native 4D 动作。

可作为增强贡献：

- K-shot Rig Calibration；
- Joint Pose–Rotation Iterative Refinement；
- Continuous Twist Field；
- Camera relation self-calibration；
- Latent flow refinement；
- Image/video unified inference；
- 任意视角子集鲁棒性。

---

## 二十二、可使用的论文标题方向

- **Skeleton-Peer Multi-View Motion Capture for Diverse Human Rigs**
- **Direct Multi-View Motion Capture for Arbitrary Human Skeletons**
- **Continuous Kinematic Fields for Multi-View Human Rig Animation**
- **Rig-Aware Multi-View Motion Capture without Canonical Skeletons**
- **Beyond SMPL: Skeleton-Peer Multi-View Human Motion Capture**
- **Rotation-Aware Multi-View Motion Capture for Heterogeneous Human Rigs**
- **Camera-Optional Motion Capture for Heterogeneous Human Rigs**
- **Calibration-Free Skeleton-Peer Human Motion Capture**
- **Rig-Native Multi-View Mocap without SMPL or Mandatory Calibration**
- **Image-to-Video Unified 4D Motion Capture for Arbitrary Human Rigs**
- **Rig-Native 4D Motion Fields for Image and Video Mocap**
- **Latent Flow Refinement for Camera-Optional Human Rig Animation**

---

## 二十三、最终技术路线

```text
图片/视频观测 + DINOv3 dense tokens + 可选相机提示
单图 / 多视角图片 / 单目视频 / 多视角视频
              ↓
相机可选的视角-时间关系编码
准确相机 / 加噪相机 / 仅视角 ID / 无相机参数
              ↓
4D Rig-native 人体运动场
F(t, chain, u)
              ↓
目标 Rig Joint Queries
层级 + Rest Pose + 语义 + 链坐标
              ↓
多参考 Pose-Rotation Anchor
              ↓
逐关节、逐自由度、相机可选的视角融合
              ↓
目标骨架位置 + Swing + Twist
              ↓
Latent Flow Kinematic Refinement
ΔP + contact + uncertainty + motion hypothesis
              ↓
目标 Rig 局部旋转
              ↓
可微 FK + 多视角重投影
或无标定跨视角一致性
              ↓
位置与旋转迭代修正
```

最终研究主张：

> **人体动作不应先被压缩到 SMPL 等固定骨架，再重定向到目标 Rig；多视角捕捉也不应把外部相机标定作为唯一入口。基于 DINOv3 dense tokens 的视觉系统应把图片和视频统一为相机可选的 4D Rig-native motion field，再由各目标骨架以平级查询的方式，直接解码自己的位置、twist 分布、接触状态和局部旋转。**
