# 课堂视觉数据处理台

这是一个本地 Streamlit 页面，用于从普通图片文件夹或 ZIP 压缩包中扫描图片和视频，分别抽样，生成低分辨率、恢复分辨率、随机黑色像素和黑色矩形遮挡版本，并导出分开的图片/视频清单。

## 启动

```powershell
cd D:\dataset_augmenter
.\install_dependencies.bat
```

安装完成后，也可以右键使用 PowerShell 运行 `run_app.ps1`。

## 桌面点击启动

第一次运行：

```powershell
cd D:\dataset_augmenter
powershell -ExecutionPolicy Bypass -File .\create_desktop_shortcut.ps1
```

之后双击桌面的“课堂视觉数据处理台”即可自动启动服务并打开浏览器。应用会在 `D:\dataset_augmenter\logs` 保存启动日志。

如果换到另一台电脑，先双击 `install_dependencies.bat` 安装依赖；需要停止后台服务时运行 `stop_app.ps1`。

## 数据安全

- 原始文件只读，不会被覆盖。
- 左侧“生成内容保存路径”支持手动输入或选择本机文件夹；图片、视频、Excel、质量检查、跟踪和模型评估结果都会使用该路径下的运行子目录。
- ZIP 内的图片可以直接扫描和处理，无需先把全部压缩包解压。
- 处理结果按运行时间写入输出目录，并生成 `sample_manifest.xlsx` 和 `sample_manifest.csv`。
- Excel 中同时包含 `sample_manifest`（一张图片一行）和 `object_manifest`（一个目标一行），适合多目标课堂行为数据。
- 多个数据源可以在页面中逐行填写，格式为：`本地路径|||数据集名称|||原始来源|||来源链接`；可参考 `source_registry_template.csv`。
- 勾选“扫描视频文件（可抽帧）”后，可以设置抽帧间隔和每个视频的最大帧数；视频帧会进入同一套图像处理流程，结果写入 `videos` 子目录，图片写入 `images` 子目录。
- 抽样结果可以按文件名/路径手动搜索并勾选，也可以按图片和视频分别查看。
- 低分辨率版本可以保留为真实小尺寸，也可以用双线性、双三次或 Lanczos 插值恢复到原尺寸；Excel 会同时记录模拟分辨率和最终输出分辨率。
- 批量预处理默认使用 JPG，以减少写盘时间和磁盘占用；需要无损像素时可在页面中切换为 PNG。处理条件数量等于“分辨率数量 × 黑色像素比例数量 × 遮挡比例数量”，建议先用少量组合试跑。
- 图片处理会复用同一帧的降分辨率结果，并降低进度条刷新频率；同一张图仍然选择很多“分辨率×噪声×遮挡”组合时，输出文件数量和磁盘写入仍会明显增加。卡顿时先用 1 个分辨率、1 个噪声档位和 1 个遮挡档位做小样本验证。
- 黑色像素和黑色方块遮挡提供0.1%、0.25%、0.5%、1%、2%等小面积档位；界面按远距离多人、普通距离、近距离目标给出建议范围。建议范围是本课题的实验起点，不是通用真实遮挡标准。
- 多人图片可以按 XML/VOC 标注框选择一个人物，也可以输入归一化 ROI；ROI 会在图片和视频帧处理前应用。
- 结构化遮挡支持标注框/ROI区域、前景学生合成遮挡和手动关键点区域遮挡；没有真实前景图时会在清单中明确标记为合成遮挡，不会伪装成真实遮挡。
- 隐私保护支持人脸模糊、马赛克、黑色遮挡，以及人体区域匿名化；人脸检测失败时会记录失败原因和回退区域。
- 视频可以固定同一处理条件下的噪声掩码和遮挡矩形，减少帧间闪烁，并可把处理帧重新合成为 MP4；跟踪结果还会输出每个 `person_id` 的观测持续时间。
- 清单增加 DAiSEE、SCB、Tianchi 和微表情数据集的任务/标签命名空间，保留原始标签和限定标签，不跨数据集强行合并。
- `impact_summary` 先统计处理条件的样本占比。真实模型评估页支持 `task,condition,metric,value` 汇总格式、`sample_id,condition,task,y_true,y_pred` 逐样本格式，以及旧版 `sample_id,baseline_score,condition_score` 格式；不会把图像变化伪装成识别结果。
- 模型评估结果必须来自真实模型在原图和处理后图像上的推理。程序只负责按 `sample_id` 对齐并汇总，不会根据像素变化猜测识别性能。
- 模型评估 CSV 可以包含 `task`、`metric`、`label`、`split`、`person_id` 等列。分类任务建议导出每个样本的正确性/得分或统一的 F1、Accuracy 等分数；回归任务导出 MAE/RMSE（数值越低越好时请在界面中选择相应方向）。同一批样本的原图与处理条件必须使用稳定且唯一的 `sample_id`。
- 发行包中的 `model_evaluation_template.csv` 使用“汇总格式”：每个任务和指标分别写入“原图”和处理条件两行，列为 `task、condition、metric、value、sample_count`；如果要按样本重新计算指标，则改用 `sample_id、condition、y_true、y_pred` 格式。
- 评估结果会按任务和处理条件汇总，输出 Accuracy、Precision、Recall、UAR、Macro-F1，投入度回归的 MAE、RMSE，以及可直接导入的 mAP、AP、AR、PCK、OKS；同时输出相对基线的变化和影响占比。不同任务或不同指标不要混在同一列中比较。
- 自动模型评估接口使用外部命令模板（例如 `python my_model.py --manifest {manifest_path} --output {output_csv}`），程序会生成 `model_input_manifest.csv`，模型写出 `model_predictions.csv` 后自动计算指标；命令使用 `shell=False`，只允许固定占位符。
- 模型插件注册表提供 YOLO 检测、YOLO 姿态、外部分类和外部回归的协议、权重路径/URL、版本、许可证和 SHA-256 字段；大型权重不随程序强制打包。可参考 `model_plugin_template.py` 编写 `predict_batch(frame, config)` 插件。
- 检测/姿态页可以从预测框和真实框自动计算 Precision、Recall、mAP@0.5，以及从关键点计算 PCK、OKS、简化 AP/AR；这些数值仍必须基于真实标注和真实模型预测。
- 指标计算支持 mAP@0.5:0.95、每类别结果、PCK@0.05/PCK@0.1 和可选 Bootstrap 置信区间；当前检测/姿态公式是单图像/单 IoU 阈值的可复核实现，正式论文结果建议与 COCO 官方评估器交叉验证。
- “⑥ 数据质量”页提供 VOC XML、YOLO TXT、COCO JSON 和关键点 JSON 的统一标注转换；跟踪页可上传 Ground Truth 轨迹计算 IDF1、MOTA、简化 HOTA、ID Switch 和遮挡恢复率。
- 报告页支持按条件生成交互曲线；安装 `requirements-visualization.txt` 后还能生成 PNG 曲线和影响柱状图。
- “⑥ 数据质量”页支持只读检查损坏文件、分辨率异常、标签缺失、重复内容和 train/test 划分泄漏，并输出 CSV/Excel 报告。
- 处理运行会写入 `checkpoint.json`；相同参数的运行可以选择断点续跑，已存在且有效的输出会被跳过。跟踪清单增加 `track_quality`、`association_iou`、`track_age`、`is_predicted`，汇总中记录重新捕获次数和最大丢失帧数。
- 每次处理运行会生成 `reproducibility.json`、`experiment_config.json` 和 `output_hashes.csv`，记录程序版本、运行时间、随机种子、输入资产签名、参数配置、模型元数据和输出文件 SHA-256；模型版本需在真实模型评估阶段补充。
- 多人跟踪功能会在视频帧中为检测到的人分配 `person_id`，并输出每帧的框、置信度和跟踪状态。跟踪 ID 只在同一视频、同一次运行内有效，不能直接当作跨视频或跨数据集的真实身份。
- 跟踪需要额外的模型依赖和权重文件；如果未安装可选依赖或未提供权重，程序仍可执行扫描、抽样和图像处理，但会明确显示跟踪不可用，不会伪造 person_id。
- 遮挡参数说明参考 Random Erasing（AAAI 2020）的面积比例设计，并结合 KeepAugment 对关键信息保护的结论；程序会把依据说明写入运行设置。
- “按子文件夹均衡”会按照根目录下的一级子文件夹抽样；如果输入的是 ZIP，则按 ZIP 文件名作为分组。

## 汇报三表

扫描数据后打开“④ 汇报三表”：

- `dataset_master`：每个公开数据集一行，登记来源、许可、模态、分辨率、标注和降质评测能力；
- `sample_index`：每个数据集默认抽取 500 个实验样本，按媒体类型、类别、视角/来源和分辨率等分层；
- `raw_file_list`：递归记录全量文件和 ZIP 成员，不把全量图片逐条展开成实验样本。

三个表会同时写入 `dataset_report.xlsx` 和独立 CSV 文件。
- 先抽样 20～100 张验证流程，再扩展到全量数据。

## 说明

降分辨率默认采用“缩小后恢复到原尺寸”的模拟方式，也可以选择保留低分辨率输出。黑色像素、黑色方框、结构化遮挡和隐私处理都是可复现的处理，参数会写入清单。原始图片是否存在真实遮挡不会自动推断，会标记为“未自动判定”。视频默认以抽取帧的形式处理，开启合成开关后会额外生成处理后 MP4，不会覆盖原始视频。

## 网盘分发与安装

上传时只需要上传本发行包 ZIP，不要把 `.venv`、`logs`、`__pycache__`、备份文件或本机实验输出一起上传。

使用者下载 ZIP 后：

1. 右键 ZIP，选择“解压到当前文件夹”或“全部解压缩”。
2. 进入解压后的 `dataset_augmenter` 文件夹。
3. 双击 `install_dependencies.bat`。脚本会在程序目录创建 `.venv` 并安装依赖。
4. 安装完成后，双击桌面上的“课堂视觉数据处理台”快捷方式。
5. 如果没有自动创建快捷方式，可右键运行 `create_desktop_shortcut.ps1`。

首次安装需要联网下载 Python 包。程序不会自带课堂数据集，使用者需要在页面中填写自己的数据目录。

## 模型评估与多人跟踪的使用顺序

建议先用少量样本验证整个链路，再扩大到完整数据集：

1. 运行一次原图和处理条件，保留生成的 `sample_manifest.xlsx`，其中的 `sample_id` 作为后续对齐键。
2. 用同一个行为、微表情、投入度或姿态模型分别推理原图和处理后的图片/视频帧，导出包含 `sample_id`、`baseline_score`、`condition_score` 的 CSV；每个任务和指标单独导出一份更容易核查。
3. 在“汇报三表”页上传 CSV，生成模型性能影响结果。分类任务常用 Accuracy、Macro-F1、UAR、mAP；回归任务常用 MAE、RMSE；姿态任务常用 AP、AR、PCK、OKS。指标方向必须保持一致，越高越好和越低越好的指标不要混用。
4. 对视频启用多人检测/跟踪后，检查跟踪预览和 `person_tracking_manifest.csv`。重点检查遮挡、交叉和出画重入时的 ID 是否发生切换；必要时降低检测阈值或改用人工 ROI 复核。

## 依赖和权重说明

基础安装只包含扫描、抽样、图像处理、清单和模型结果汇总所需依赖。多人检测/跟踪模型通常需要额外安装推理库并下载模型权重；权重文件的许可和来源必须单独记录，不能把它们和本程序的代码许可混为一谈。网盘分发时建议把权重放在独立的 `weights` 文件夹，并在 README 中写明文件名、来源 URL、版本和 SHA-256。

若要启用 YOLO/ByteTrack，在完成基础安装后运行 `install_tracking_dependencies.bat`。这个可选安装会下载较大的 `ultralytics` 和其深度学习依赖；首次选择模型时还可能联网下载 `yolo11n.pt` 权重。未安装时，基础处理和模型评估导入仍可用。

## EXE 版本

发行包同时提供基础 EXE 版本。双击 `课堂视觉数据处理台.exe` 即可启动网页，不需要单独安装 Python；该版本包含扫描、抽样、图像/视频处理和真实模型评估导入。由于 YOLO/ByteTrack 需要较大的 PyTorch 依赖和模型权重，EXE 版本不内置这部分，使用多人跟踪时请使用完整 ZIP 并运行 `install_tracking_dependencies.bat`。
